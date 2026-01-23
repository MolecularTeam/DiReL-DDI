import torch
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import warnings

from data_preprocessing_i import DrugDataset, DrugDataLoader
from get_args import get_config
from models_v5 import DiffusionDDIModel

warnings.filterwarnings('ignore', category=UserWarning)
test_dataset_name = "miner"  # miner, ddinter, drugbank

######################### Config ########################
cfg = get_config()

cuda_num = cfg.cuda_num
params = cfg.params
rank = cfg.params.rank
batch_size = params.batch_size
neg_samples = params.neg_samples
data_size_ratio = params.data_size_ratio
device = f"cuda:{cuda_num}" if (torch.cuda.is_available() and params.use_cuda) else "cpu"

# dataset
n_atom_feats = 55
rel_total = 86
hidd_dim = 128
kge_dim = 128

print(f"[Dataset] {test_dataset_name}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, f"pkl_{test_dataset_name}")


def make_pkl_name(fold, s, dataset_name):
    return f"induc_fold{fold}_{s}_{dataset_name}.pkl"

def load_induc_dataset(dataset_name, fold, split):
    if 'drugbank' in dataset_name or 'ddinter' in dataset_name:
        df = pd.read_csv(f"./dataset/{dataset_name}/fold{fold}/{split}.csv")
    else:
        df = pd.read_csv(f"./dataset/{dataset_name}/fold{fold}/test_S{split[-1]}.csv")

    if 'miner' in dataset_name or 'ddinter' in dataset_name:
        tup = [(h, t, r) for h, t, r in zip(df['drugbank_id_1'], df['drugbank_id_2'], df['label'])]
    else:
        tup = [(h, t, r) for h, t, r in zip(df['d1'], df['d2'], df['type'])]

    return tup

def do_compute(batch, device, model):
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)

    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)

    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)
    return acc, auroc, f1_score, precision, recall, int_ap, ap

def test(test_loader, model):
    probas_pred_all = []
    ground_truth_all = []

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            _, _, probas_pred, ground_truth = do_compute(batch, device, model)
            probas_pred_all.append(probas_pred)
            ground_truth_all.append(ground_truth)

        probas_pred_all = np.concatenate(probas_pred_all)
        ground_truth_all = np.concatenate(ground_truth_all)
        acc, auc, f1, precision, recall, int_ap, ap = \
            do_compute_metrics(probas_pred_all, ground_truth_all)


    print("====================== Final Best Result ======================")
    print(f" {split} → acc:{acc:.4f}, auc:{auc:.4f}, f1:{f1:.4f}, pre:{precision:.4f}, rec:{recall:.4f}, intAP:{int_ap:.4f}, AP:{ap:.4f}")

    return {
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "int_ap": int_ap,
        "ap": ap
    }


if __name__ == "__main__":
    print("\n📌 Building DiffusionDDIModel...")

    encoder_cfg = dict(
        in_node_features=[n_atom_feats, 2048, 200],
        in_edge_features=17,
        hidd_dim=hidd_dim,
        kge_dim=kge_dim,
        rel_total=rel_total,
        heads_out_feat_params=[32,32,32,32],
        blocks_params=[4,4,4,4],
        edge_feature=64,
        dp=0.06
    )

    head_cfg = {"rel_total": rel_total, "dim": kge_dim, "rank": rank, "noise": 0.05}
    all_results = []

    if 'ddinter' not in test_dataset_name:
        folds = [0, 1, 2]
    else:
        folds = [0, 1, 2, 3, 4]

    for split in ["s1", "s2"]:
        for fold in folds:
            print(f"\n================ FOLD {fold} ================")
            model = DiffusionDDIModel(encoder_cfg, head_cfg).to(device)

            pkl_name = make_pkl_name(fold, split, test_dataset_name)
            pkl_path = os.path.join(PKL_DIR, pkl_name)

            print(f"▶ pkl name split={pkl_name}")
            state = torch.load(pkl_path, map_location=device, weights_only=True)
            model.load_state_dict(state)

            print(f"\n▶ Evaluating split={split}")
            data_tup = load_induc_dataset(test_dataset_name, fold, split)
            data = DrugDataset(data_tup, disjoint_split=True)
            test_loader = DrugDataLoader(data, batch_size=batch_size * 3, shuffle=False, num_workers=2)

            result = test(test_loader, model)

            all_results.append({
                "fold": fold,
                "split": split,
                **result
            })

    df = pd.DataFrame(all_results)
    df_all = df.copy()

    mean_df = df.groupby("split").mean(numeric_only=True).reset_index()

    # metric
    std_metrics = ["acc", "auc", "f1", "ap"]

    # std
    std_df = (
        df.groupby("split")[std_metrics]
        .std()
        .add_suffix("_std")
        .reset_index()
    )

    # mean + std
    summary_df = pd.merge(mean_df, std_df, on="split")

    summary_df["fold"] = "mean"
    summary_df["seed"] = "mean"
    summary_df["split"] = summary_df["split"] + "_mean"
    summary_df = summary_df[df_all.columns.tolist() + [f"{m}_std" for m in std_metrics]]


    for m in ["acc", "auc", "f1", "ap"]:
        summary_df[m.upper()] = summary_df.apply(
            lambda r: f"{r[m] * 100:.2f} ± {r[f'{m}_std'] * 100:.2f}", axis=1
        )
    final_df = pd.concat([df_all, summary_df], ignore_index=True)

    final_df.to_csv(f"{test_dataset_name}_final_results_with_mean.csv", index=False)
    print(final_df)




