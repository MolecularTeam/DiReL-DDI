# **DiReL-DDI: Diffusion-Regularized Dynamic Relational Learning for Drug-Drug Interaction Prediction**

This is the **official** PyTorch implementation of  
**"DiReL-DDI: Diffusion-Regularized Dynamic Relational Learning for Drug–Drug Interaction Prediction"**.

Our PyTorch implementation is based on several prior DDI works:

- **SSI-DDI**: Substructure–Substructure Interactions for Drug–Drug Interaction Prediction  
  https://github.com/kanz76/SSI-DDI

- **DSN-DDI**: An Accurate and Generalized Framework for Drug–Drug Interaction Prediction  
  https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction

- **PEB-DDI**: A Task-Specific Dual-View Substructural Learning Framework  
  https://github.com/wayyzt/PEB-DDI

---

## Abstract

**Motivation.** Polypharmacy is increasingly prevalent in clinical practice, substantially elevating the risk of drug–drug interactions (DDIs). Accurate computational DDI prediction is therefore essential for improving drug safety.  
Although recent deep learning–based models have advanced molecular representation learning, many existing DDI models represent drug interactions at the level of interaction types, without sufficiently reflecting the diversity of interaction patterns among individual drug pairs within the same DDI type. This limitation highlights the importance of more flexible approaches that allow individual drug pairs to be represented in a pair-specific manner.

**Results.** We propose DiReL-DDI, a diffusion-regularized dynamic relational learning framework for DDI prediction, which models drug interactions through pair-specific interaction representations.  
To improve the stability of interaction modeling, DiReL-DDI incorporates diffusion-based score matching as a regularization component. Experiments on benchmark datasets demonstrate that DiReL-DDI exhibits superior performance, supported by comprehensive analyses of the proposed framework.

---

## Requirements (Our Environment)

- Python ≥ 3.10  
- PyTorch ≥ 2.5.1  
- CUDA (PyTorch): 12.1  
- cuDNN: 9.1.0  
- torch-geometric: 2.6.1  

---

## Dataset

The datasets used in our experiments are provided in the `dataset/` directory.
Please refer to the files in this directory for details.

---

## Usage

The training and evaluation setup follows prior work, with evaluation conducted on the inductive settings.

To train the model, run:
```bash
python main.py
```

To evaluate a trained model on the inductive settings, run:
```bash
python main_test.py
```

