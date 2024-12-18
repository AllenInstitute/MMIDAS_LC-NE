import numpy as np
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
from scipy.sparse import issparse


def normalize_cellxgene(x):
    """Normalize based on number of input genes

    Args:
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
        scale_factor (float): Scalar multiplier
    
    Returns: 
        x, np.mean(x)
    """
    # x = np.divide(x, np.sum(x, axis=1, keepdims=True))*scale_factor

    return normalize(x, axis=1, norm='l1')


def logcpm(x, scaler=1e4) -> np.array:
    """ Log CPM normalization

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
        scaler (float, optional): scaling factor for log CPM
    
    return 
        normalized log CPM gene expression matrix
    """
    return np.log1p(normalize_cellxgene(x) * scaler)


def sparse_std(x, axis=None):
    x_ = x.copy()
    x_.data **= 2
    return np.sqrt(x_.mean(axis) - np.square(x.mean(axis)))


def reorder_genes(x, eps, chunksize=1000):
    t_gene = x.shape[1]
    g_std = []
    if issparse(x):
        g_std = sparse_std(x, axis=0)
    else:
        g_std = np.std(x, axis=0)

    # for iter in range(int(t_gene // chunksize) + 1):
    #     ind0 = iter * chunksize
    #     ind1 = np.min((t_gene, (iter + 1) * chunksize))
    #     # x_bin = np.where(x[:, ind0:ind1] > eps, 1, 0)
    #     g_std.append(np.std(x[:, ind0:ind1], axis=0))
    #     g_bin_std.append(np.std(x_bin, axis=0))

    # g_std = np.concatenate(g_std)
    # g_bin_std = np.concatenate(g_bin_std)
    g_ind = np.argsort(g_std)
    g_ind = g_ind[np.sort(g_std) > eps]
    return g_ind[::-1]


def get_HVG(x, thr=0.1, binary=True):
    if binary:
        x_bin = (x != 0).astype(int)
        g_index = reorder_genes(x_bin, thr)
    else:
        g_index = reorder_genes(x, thr)
    
    return np.asarray(g_index).flatten()


def tfidf(x, scaler=1e4):
    tfidf = TfidfTransformer(norm='l1', use_idf=True)
    return tfidf.fit_transform(x) * scaler

def get_HAP(x, thr=0.1, binary=True):
    if binary:
        x_bin = (x != 0).astype(int)
        colsum = np.array(np.sum(x_bin, axis=0))
    else:
        colsum = np.array(np.sum(x, axis=0))

    colsum = colsum.reshape(-1)
    return np.logical_and((colsum > x.shape[0] * thr), (colsum < x.shape[0] * (1 - thr)))


def split_data_Kfold(class_label, K_fold):
    uniq_label = np.unique(class_label)
    label_train_indices = [[] for ll in uniq_label]
    label_test_indices = [[] for ll in uniq_label]

    # Split the the data to train and test keeping the same ratio for all classes
    for i_l, label in enumerate(uniq_label):
        label_indices = np.where(class_label == label)[0]
        test_size = int(( 1 /K_fold) * len(label_indices))

        # Prepare the test and training indices for K folds
        for fold in range(K_fold):
            ind_0 = fold * test_size
            ind_1 = (1 + fold) * test_size
            tmp_ind = list(label_indices)
            label_test_indices[i_l].append(tmp_ind[ind_0:ind_1])
            del tmp_ind[ind_0:ind_1]
            label_train_indices[i_l].append(tmp_ind)
    test_ind = [[] for k in range(K_fold)]
    train_ind = [[] for k in range(K_fold)]
    for fold in range(K_fold):
        for i_l in range(len(uniq_label)):
            test_ind[fold].append(label_test_indices[i_l][fold])
            train_ind[fold].append(label_train_indices[i_l][fold])
        test_ind[fold] = np.concatenate(test_ind[fold])
        train_ind[fold] = np.concatenate(train_ind[fold])
        # Shuffle the indices
        index = np.arange(len(test_ind[fold]))
        np.random.shuffle(index)
        test_ind[fold] = test_ind[fold][index]
        index = np.arange(len(train_ind[fold]))
        np.random.shuffle(index)
        train_ind[fold] = train_ind[fold][index]

    return train_ind, test_ind



