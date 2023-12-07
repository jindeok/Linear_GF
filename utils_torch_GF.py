
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.stats import rankdata
import scipy.sparse as sp
import copy
from scipy.linalg import expm



def csr2torch(csr_matrix):
    # Convert CSR matrix to COO format (Coordinate List)
    coo_matrix = csr_matrix.tocoo()

    # Create a PyTorch tensor for data, row indices, and column indices
    data = torch.FloatTensor(coo_matrix.data)
    indices = torch.LongTensor([coo_matrix.row, coo_matrix.col])

    # Create a sparse tensor using torch.sparse
    return torch.sparse.FloatTensor(indices, data, torch.Size(coo_matrix.shape))


def normalize_sparse_adjacency_matrix(adj_matrix, alpha):
    # Calculate rowsum and columnsum using COO format
    rowsum = torch.sparse.mm(adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)).squeeze()
    rowsum = torch.pow(rowsum, -alpha)
    colsum = torch.sparse.mm(adj_matrix.t(), torch.ones((adj_matrix.shape[0], 1), device=adj_matrix.device)).squeeze()
    colsum = torch.pow(colsum, alpha-1)
    indices = torch.arange(0, rowsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    d_mat_rows = torch.sparse.FloatTensor(indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])).to(device=adj_matrix.device)
    indices = torch.arange(0, colsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    d_mat_cols= torch.sparse.FloatTensor(indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])).to(device=adj_matrix.device)
    # Normalize adjacency matrix
    norm_adj = d_mat_rows.mm(adj_matrix).mm(d_mat_cols)

    return norm_adj
def recall_at_k(gt_mat, results, k=10):
    recall_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        top_predicted_items = np.argsort(-results[i, :])[:k]
        num_relevant_items = len(relevant_items.intersection(top_predicted_items))
        recall_sum += num_relevant_items / len(relevant_items)
    recall = recall_sum / gt_mat.shape[0]
    return recall

def ndcg_at_k(gt_mat, results, k=10):
   ndcg_sum = 0
   for i in range(gt_mat.shape[0]):
       relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
       top_predicted_items = np.argsort(-results[i, :])[:k]
       dcg = 0
       idcg = 0
       for j in range(k):
           if top_predicted_items[j] in relevant_items:
               dcg += 1 / np.log2(j + 2)
           if j < len(relevant_items):
               idcg += 1 / np.log2(j + 2)
       ndcg_sum += dcg / idcg if idcg > 0 else 0
   ndcg = ndcg_sum / gt_mat.shape[0]
   return ndcg

def recall_at_k(gt_mat, results, k=10):
    recall_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        top_predicted_items = np.argsort(-results[i, :])[:k]
        num_relevant_items = len(relevant_items.intersection(top_predicted_items))
        recall_sum += num_relevant_items / len(relevant_items)
    recall = recall_sum / gt_mat.shape[0]
    return recall

def ndcg_at_k(gt_mat, results, k=10):
   ndcg_sum = 0
   for i in range(gt_mat.shape[0]):
       relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
       top_predicted_items = np.argsort(-results[i, :])[:k]
       dcg = 0
       idcg = 0
       for j in range(k):
           if top_predicted_items[j] in relevant_items:
               dcg += 1 / np.log2(j + 2)
           if j < len(relevant_items):
               idcg += 1 / np.log2(j + 2)
       ndcg_sum += dcg / idcg if idcg > 0 else 0
   ndcg = ndcg_sum / gt_mat.shape[0]
   return ndcg

def top_k(S, k=1, device = 'cpu'):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    if device == 'cpu':
        top = np.argsort(-S)[:, :k]
        result = np.zeros(S.shape)
        for idx, target_elms in enumerate(top):
            for elm in target_elms:
                result[idx, elm] = 1
    else:
        top = torch.argsort(-S)[:, :k]
        result = torch.zeros(S.shape, device =device)
        for idx, target_elms in enumerate(top):
            for elm in target_elms:
                result[idx, elm] = 1
    return result, top
    