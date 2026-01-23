import torch
from torch import optim
import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime
from sklearn import metrics
import warnings
from data_preprocessing_i import DrugDataset, DrugDataLoader

from get_args import get_config
from utils import ensure_dir

from models_v5 import DiffusionDDIModel
from custom_loss import SigmoidLoss, DiffusionLoss

warnings.filterwarnings('ignore', category=UserWarning)

######################### Config ########################
cfg = get_config()

cuda_num = cfg.cuda_num
dataset_name = cfg.dataset_name
dataset_cfg = cfg[dataset_name]
fold = cfg.dataset_name[-1]

params = cfg.params
lr = params.lr
n_epochs = params.n_epochs
batch_size = params.batch_size
rank = params.rank
weight_decay = params.weight_decay
neg_samples = params.neg_samples
T_max = params.T_max
data_size_ratio = params.data_size_ratio
device = f"cuda:{cuda_num}" if (torch.cuda.is_available() and params.use_cuda) else "cpu"

original_dataset_name = dataset_name.split('_')[0]
pkl_filename = os.path.basename(cfg[dataset_name]["inductive_pkl_dir"])

log_path = f"./logs_{original_dataset_name}"
pkl_path = f"./pkl_{original_dataset_name}"

pkl_name = os.path.join(pkl_path, pkl_filename)

ensure_dir(log_path)
ensure_dir(pkl_path)

n_atom_feats = 55
rel_total = 86
hidd_dim = 128
kge_dim = 128

print(f"[Dataset] {dataset_name}")
print(params)

train_file_name = f"training_{dataset_name}.csv"
test_file_name  = f"test_{dataset_name}.csv"

########## Dataset loader ##########
if 'miner' in dataset_name or 'ddinter' in dataset_name:
    df_ddi_train = pd.read_csv(dataset_cfg.induc_ddi_train)
    df_ddi_s1 = pd.read_csv(dataset_cfg.induc_s1)
    df_ddi_s2 = pd.read_csv(dataset_cfg.induc_s2)

    train_tup = [(h,t,r) for h,t,r in zip(df_ddi_train['drugbank_id_1'],df_ddi_train['drugbank_id_2'],df_ddi_train['label'])]
    s1_tup    = [(h,t,r) for h,t,r in zip(df_ddi_s1['drugbank_id_1'], df_ddi_s1['drugbank_id_2'], df_ddi_s1['label'])]
    s2_tup    = [(h,t,r) for h,t,r in zip(df_ddi_s2['drugbank_id_1'], df_ddi_s2['drugbank_id_2'], df_ddi_s2['label'])]

else:
    df_ddi_train = pd.read_csv(dataset_cfg.induc_ddi_train)
    df_ddi_s1 = pd.read_csv(dataset_cfg.induc_s1)
    df_ddi_s2 = pd.read_csv(dataset_cfg.induc_s2)

    train_tup = [(h,t,r) for h,t,r in zip(df_ddi_train['d1'],df_ddi_train['d2'],df_ddi_train['type'])]
    s1_tup    = [(h,t,r) for h,t,r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
    s2_tup    = [(h,t,r) for h,t,r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

# dataset
train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
s1_data    = DrugDataset(s1_tup, disjoint_split=True)
s2_data    = DrugDataset(s2_tup, disjoint_split=True)

print(f"Train={len(train_data)}  S1={len(s1_data)}  S2={len(s2_data)}")

train_loader = DrugDataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=2)
s1_loader    = DrugDataLoader(s1_data,batch_size=batch_size*3,num_workers=2)
s2_loader    = DrugDataLoader(s2_data,batch_size=batch_size*3,num_workers=2)


def summarize_model(model):
    print("\n================ Model Summary ================")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print("===============================================\n")


def get_lambda(epoch, max_epoch, max_lambda=0.2):
    if epoch < max_epoch * 0.3:
        return max_lambda * (epoch / (max_epoch * 0.3))
    elif epoch < max_epoch * 0.7:
        return max_lambda
    else:
        decay_ratio = 1 - (epoch - max_epoch * 0.7) / (max_epoch * 0.3)
        decay_ratio = max(decay_ratio, 0)
        return max_lambda * decay_ratio


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


def train(model, train_data_loader, s1_data_loader, s2_data_loader,
          loss_fn, diff_loss_fn, optimizer, n_epochs, device, scheduler=None):

    print('Starting training at', datetime.today())
    s1_acc_max = -1.0
    s2_acc_max = -1.0
    logs = []

    early_epoch = 10
    early_stop_cnt = 0

    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0.0
        s1_loss = 0.0
        s2_loss = 0.0

        train_probas_pred = []
        train_ground_truth = []

        s1_probas_pred = []
        s1_ground_truth = []
        s2_probas_pred = []
        s2_ground_truth = []

        lambda_diff = get_lambda(i, n_epochs)


        for batch in train_data_loader:
            model.train()
            pos_tri, neg_tri = batch

            pos_tri = [tensor.to(device=device) for tensor in pos_tri]
            neg_tri = [tensor.to(device=device) for tensor in neg_tri]

            p_score, p_eps_pred, p_eps_true = model(pos_tri, return_loss=True)
            n_score, n_eps_pred, n_eps_true = model(neg_tri, return_loss=True)

            base_loss = loss_fn(p_score, n_score)
            diff_loss = (diff_loss_fn(p_eps_pred, p_eps_true) + diff_loss_fn(n_eps_pred, n_eps_true)) / 2

            loss = base_loss + lambda_diff * diff_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)

        train_loss /= len(train_data)

        with torch.no_grad():
            for batch in train_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                train_probas_pred.append(probas_pred)
                train_ground_truth.append(ground_truth)

            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision, train_recall, train_int_ap, train_ap = \
                do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in s1_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s1_probas_pred.append(probas_pred)
                s1_ground_truth.append(ground_truth)
                loss_eval = loss_fn(p_score, n_score)
                s1_loss += loss_eval.item() * len(p_score)

            s1_loss /= len(s1_data)
            s1_probas_pred = np.concatenate(s1_probas_pred)
            s1_ground_truth = np.concatenate(s1_ground_truth)
            s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = \
                do_compute_metrics(s1_probas_pred, s1_ground_truth)

            for batch in s2_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s2_probas_pred.append(probas_pred)
                s2_ground_truth.append(ground_truth)
                loss_eval = loss_fn(p_score, n_score)
                s2_loss += loss_eval.item() * len(p_score)

            s2_loss /= len(s2_data)
            s2_probas_pred = np.concatenate(s2_probas_pred)
            s2_ground_truth = np.concatenate(s2_ground_truth)
            s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = \
                do_compute_metrics(s2_probas_pred, s2_ground_truth)

            if s1_acc > s1_acc_max:
                s1_acc_max = s1_acc
                early_stop_cnt = 0
                torch.save(model.state_dict(), pkl_name)
            # if s2_acc>s2_acc_max:
            #     s2_acc_max = s2_acc
            #     torch.save(model,pkl_name)

            else:
                early_stop_cnt += 1

            print(
                f"[Epoch {i}] "
                f"acc={s1_acc:.4f} | "
                f"best={s1_acc_max:.4f} | "
                f"early_stop={early_stop_cnt}/{early_epoch}"
            )

            if early_stop_cnt >= early_epoch:
                print(f"Early stopping triggered at epoch {i}")
                break

        if scheduler:
            scheduler.step()

        print(f'Epoch: {i} ({time.time() - start:.4f}s)'
              f' | total={train_loss:.4f}  base={base_loss.item():.4f}  diff={diff_loss.item():.4f}'
              f' | λ={lambda_diff:.3f}')
        print(f'\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f}, '
              f'precision:{train_precision:.4f}, recall:{train_recall:.4f}')
        print(f'\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, '
              f'precision:{s1_precision:.4f}, recall:{s1_recall:.4f}')
        print(f'\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, '
              f'precision:{s2_precision:.4f}, recall:{s2_recall:.4f}')

        logs.append({
            'epoch': i,
            'train_loss': train_loss,
            'base_loss': base_loss.item(),
            'diff_loss': diff_loss.item(),
            'lambda': lambda_diff,
            's1_loss': s1_loss,
            's2_loss': s2_loss,
            'train_acc': train_acc,
            'train_roc': train_auc_roc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            's1_acc': s1_acc,
            's1_roc': s1_auc_roc,
            's2_acc': s2_acc,
            's2_roc': s2_auc_roc,
        })

        if log_path:
            pd.DataFrame(logs).to_csv(f"{log_path}/{train_file_name}", index=False)

    print("Training complete.")


def test(s1_loader, s2_loader, model):
    s1_probas_pred = []
    s1_ground_truth = []
    s2_probas_pred = []
    s2_ground_truth = []

    with torch.no_grad():
        for batch in s1_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            s1_probas_pred.append(probas_pred)
            s1_ground_truth.append(ground_truth)

        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        s1_acc, s1_auc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = \
            do_compute_metrics(s1_probas_pred, s1_ground_truth)

        for batch in s2_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            s2_probas_pred.append(probas_pred)
            s2_ground_truth.append(ground_truth)

        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        s2_acc, s2_auc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = \
            do_compute_metrics(s2_probas_pred, s2_ground_truth)

    print("\n====================== Final Best Result ======================")
    print(f" S1 → acc:{s1_acc:.4f}, auc:{s1_auc:.4f}, f1:{s1_f1:.4f}, pre:{s1_precision:.4f}, rec:{s1_recall:.4f}, intAP:{s1_int_ap:.4f}, AP:{s1_ap:.4f}")
    print(f" S2 → acc:{s2_acc:.4f}, auc:{s2_auc:.4f}, f1:{s2_f1:.4f}, pre:{s2_precision:.4f}, rec:{s2_recall:.4f}, intAP:{s2_int_ap:.4f}, AP:{s2_ap:.4f}")

    result = {
        'model_name':[pkl_name],
        's1_acc':[s1_acc], 's1_auc':[s1_auc], 's1_f1':[s1_f1],
        's1_precision':[s1_precision], 's1_recall':[s1_recall],
        's1_int_ap':[s1_int_ap], 's1_ap':[s1_ap],
        's2_acc':[s2_acc], 's2_auc':[s2_auc], 's2_f1':[s2_f1],
        's2_precision':[s2_precision], 's2_recall':[s2_recall],
        's2_int_ap':[s2_int_ap], 's2_ap':[s2_ap],
    }
    pd.DataFrame(result).to_csv(f"{log_path}/{test_file_name}", index=False)
    print(f"📄 Saved → {log_path}{test_file_name}")


######################### Run #########################
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
    model = DiffusionDDIModel(encoder_cfg, head_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    loss = SigmoidLoss()
    diff_loss = DiffusionLoss()

    train(model, train_loader, s1_loader, s2_loader, loss, diff_loss, optimizer, n_epochs, device, scheduler)
    test_model = DiffusionDDIModel(encoder_cfg, head_cfg).to(device)

    state = torch.load(f"./pkl_{original_dataset_name}/{pkl_filename}", map_location=device, weights_only=True)
    test_model.load_state_dict(state)
    test(s1_loader, s2_loader, test_model)

