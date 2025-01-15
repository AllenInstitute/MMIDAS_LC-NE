import numpy as np
import scipy.sparse as ss
import scanpy as sc
import seaborn as sns
from sklearn.preprocessing import normalize
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split



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


def generate_colors(n):
    # Generate `n` unique colors using the `viridis` colormap
    # if n <= 10:  # The colorblind palette has 10 colors
    #     palette = sns.color_palette("colorblind", n)
    # else:
    palette = sns.color_palette("hsv", n)  # Use HSV for larger n
    colors = [mcolors.rgb2hex(color) for color in palette]
    
    return np.array(colors)


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



def load_data(file, gene_file='', n_gene=0):

    adata = sc.read_h5ad(file)
    data = dict()
    data['log1p'] = adata.X.toarray()
    data['gene_id'] = adata.var.index.values
    
    if gene_file:
        df_ = pd.read_csv(gene_file)
        for key in df_.keys():
            # check key include gene in the name
            if 'gene' in key.lower():
                gene_list = df_[key].values
                break
        # search gene list in the data
        gene_index = []
        for gg in gene_list:
            if gg not in data['gene_id']:
                print(f"Gene {gg} not found in the data")
            else:
                gene_index.append(np.where(data['gene_id'] == gg)[0][0])
        data['log1p'] = data['log1p'][:, np.array(gene_index)]
        data['gene_id'] = data['gene_id'][np.array(gene_index)]
        # gene_index = [np.where(data['gene_id'] == gg)[0][0] for gg in gene_list]
        # data['log1p'] = data['log1p'][:, gene_index]
        # data['gene_id'] = data['gene_id'][gene_index]
    
    if n_gene > 0:
        data['log1p'] = data['log1p'][:, :n_gene]
        data['gene_id'] = data['gene_id'][:n_gene]
    
    for key in adata.obs.keys():
        data[key] = adata.obs[key].values
        if key == 'sex':
            data[key] = np.array([s.split(';')[0] for s in data[key]])
        
            
    print(f"Number of cells: {data['log1p'].shape[0]}, Number of genes: {data['log1p'].shape[1]}")

    return data



def get_data(x, train_size, additional_val, seed=0):

        test_size = x.shape[0] - train_size
        train_cpm, test_cpm, train_ind, test_ind = train_test_split(x, np.arange(x.shape[0]), train_size=train_size, test_size=test_size, random_state=seed)
        
        if additional_val:
            train_cpm, val_cpm, train_ind, val_ind = train_test_split(train_cpm, train_ind, train_size=train_size - test_size, test_size=test_size, random_state=seed)
        else:
            val_cpm = []
            val_ind = []

        return train_cpm, val_cpm, test_cpm, train_ind, val_ind, test_ind



def get_loaders(x, label=[], batch_size=128, train_size=0.9, n_aug_smp=0, netA=None, aug_param=0., device=None, additional_val=False):

    if len(label) > 0:
        train_ind, val_ind, test_ind = [], [], []
        for ll in np.unique(label):
            indx = np.where(label == ll)[0]
            tt_size = int(train_size * sum(label == ll))
            _, _, _, train_subind, val_subind, test_subind = get_data(x, tt_size, additional_val)
            train_ind.append(indx[train_subind])
            test_ind.append(indx[test_subind])
            if additional_val:
                val_ind.append(indx[val_subind])

        train_ind = np.concatenate(train_ind)
        test_ind = np.concatenate(test_ind)
        train_set = x[train_ind, :]
        test_set = x[test_ind, :]
        
        if additional_val:
            val_ind = np.concatenate(val_ind)
            val_set = x[val_ind, :]
        
    else:
        tt_size = int(train_size * x.shape[0])
        train_set, val_set, test_set, train_ind, val_ind, test_ind = get_data(x, tt_size, additional_val)

    train_set_torch = torch.FloatTensor(train_set)
    train_ind_torch = torch.FloatTensor(train_ind)
    if n_aug_smp > 0:
        train_set = train_set_torch.clone()
        train_set_ind = train_ind_torch.clone()
        for _ in range(n_aug_smp):
            if netA:
                noise = 0.1*torch.randn(train_set_torch.shape[0], aug_param['num_n'], device=device)
                if device:
                    _, gen_data = netA(train_set_torch.cuda(device), noise, True, device)
                else:
                    _, gen_data = netA(train_set_torch, noise, True, device)
                data_bin = 0. * train_set_torch
                data_bin[train_set_torch > 1e-4] = 1.
                fake_data = gen_data * data_bin
                train_set = torch.cat((train_set, fake_data.cpu().detach()), 0)

            else:
                train_set = torch.cat((train_set, train_set_torch), 0)
                
            train_set_ind = torch.cat((train_set_ind, train_ind_torch), 0)

        train_data = TensorDataset(train_set, train_set_ind)
    else:
        train_data = TensorDataset(train_set_torch, train_ind_torch)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    test_set_torch = torch.FloatTensor(test_set)
    test_ind_torch = torch.FloatTensor(test_ind)
    test_data = TensorDataset(test_set_torch, test_ind_torch)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)

    data_set_troch = torch.FloatTensor(x)
    all_ind_torch = torch.FloatTensor(range(x.shape[0]))
    all_data = TensorDataset(data_set_troch, all_ind_torch)
    alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    
    if additional_val:
        val_set_torch = torch.FloatTensor(val_set)
        val_ind_torch = torch.FloatTensor(val_ind)
        validation_data = TensorDataset(val_set_torch, val_ind_torch)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)
    else:
        validation_loader = []
        val_ind = []

    return alldata_loader, train_loader, test_loader, validation_loader, test_ind, val_ind




