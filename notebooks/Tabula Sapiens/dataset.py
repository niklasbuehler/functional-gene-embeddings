import scanpy as sc
import numpy as np
import polars as pl
import pandas as pd
import umap
import torch
import scipy.stats
import pyarrow

from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, random_split
import anndata

# dataset loader for the compressed TS data used for the normal VAE
class TS_Compressed_VAE_Dataset(Dataset): 
    def __init__(self,
                 path):
        self.ds_path = path

        ts_data = sc.read_h5ad(path)

        gene_list = ts_data.var.ensemblid.to_list()
        gene_list = [i.split(".")[0] for i in gene_list]
        ts_data.var_names = gene_list
        ts_data = ts_data.to_df().T
        ts_data.index.name = "gene_id"
        ts_data = pl.from_pandas(ts_data, include_index=True)

        #global normalization
        data = ts_data.melt(id_vars=['gene_id'],
                    variable_name='cell_line',
                    value_name='score',
                    )
        mean = data.select(pl.mean("score"))[0,0]
        sd = data.select(pl.std("score"))[0,0]
        ts_data = ts_data.with_columns((pl.col(ts_data.columns[1:]) - mean)/sd)

        # shuffle and split in train / test 
        ts_data = ts_data.sample(fraction = 1, shuffle = True)
        
        self.dataset = ts_data
        self.train_table = ts_data[:int(len(ts_data) * 0.9)]
        self.test_table = ts_data[int(len(ts_data) * 0.9):]

    def genes_in_assay(self):
        return self.dataset["gene_id"].to_list()
    
    def num_celllines_in_assay(self):
        return len(self.dataset.columns)-1
    
    def get_num_batches_per_epoch(self, batchsize, subset='train'):
        if subset == 'train':
            return int(np.ceil(len(self.train_table) / batchsize))
        elif subset == 'test':
            return int(np.ceil(len(self.test_table) / batchsize))
        
    def get_batches(self, batchsize, subset = 'train'):
        if subset == "train":
            table = self.train_table
        if subset == "test":
            table = self.test_table
        
        table = table.sample(fraction = 1, shuffle = True)
        # compute number of batches
        n_batches = int(np.ceil(len(table) / batchsize))
        
        start_idx = 0
        batchsize = int(batchsize)
        for batch_idx in range(n_batches):
            if batch_idx < (n_batches - 1):
                s = slice(start_idx, start_idx + batchsize, 1)
                start_idx += batchsize
            else:
                s = slice(start_idx, len(table), 1)

            subset_table = table[s]
            yield subset_table[:, 0].to_list(), torch.from_numpy(subset_table[:, 1:].to_numpy().astype('float32'))
    


# dataset class for the compressed tabula sapiens dataset used in the model by the paper
class TS_Compressed_Dataset(Dataset):
    def __init__(self,
                 gene_index,
                 path='',
                 ):

        self.ds_path = path

        ts_data = sc.read_h5ad(path)

        # processing of the dataframe
        gene_list = ts_data.var.ensemblid.to_list()
        gene_list = [i.split(".")[0] for i in gene_list]
        ts_data.var_names = gene_list
        ts_data = ts_data.to_df()
        ts_data["sample_idx"] = ts_data.index
        ts_data.sample_idx = ts_data.sample_idx.apply(lambda x: int(x))

        # reshape the dataframe so that each row contains gene index, sample index and the score
        ts_data = ts_data.melt(id_vars=["sample_idx"], var_name="gene_id", value_name="score")
        ts_data = pd.merge(ts_data, gene_index, how="inner", on="gene_id")

        # normalize the score by the global mean and variance
        self.data_mean = ts_data.score.mean()
        self.data_std = ts_data.score.std()

        ts_data.score = (ts_data.score - self.data_mean) / self.data_std

        ts_data = pl.from_pandas(ts_data)

        self.dataset = ts_data

        # shuffle and split in train / test
        self.dataset = self.dataset.sample(fraction=1, shuffle=True)

        self.train_table = self.dataset[:int(len(self.dataset) * 0.9)]
        self.test_table = self.dataset[int(len(self.dataset) * 0.9):]

    def compute_sample_init_pca(self, dim=128):
        # compute the sample pcas for the initialization of sample embeddings
        ts_data = sc.read_h5ad(self.ds_path)
        ts_data = ts_data.to_df()
        ts_data = (ts_data - self.data_mean) / self.data_std

        pca = PCA(dim)

        pcs = pca.fit_transform(ts_data)

        return pcs

    def get_num_batches_per_epoch(self, batchsize):
        return int(np.ceil(len(self.train_table) / batchsize))

    def genes_in_assay(self):
        return self.train_table.get_column("gene_id").unique()

    def get_batches(self, batchsize, subset='train'):
        """
        this function returns a generator that gives the gene indices, sample indices and scores used in each batch
        """
        if subset == "train":
            table = self.train_table
        if subset == "test":
            table = self.test_table

        table = table.sample(fraction=1, shuffle=True)
        # compute number of batches
        n_batches = int(np.ceil(len(table) / batchsize))

        start_idx = 0
        batchsize = int(batchsize)
        for batch_idx in range(n_batches):
            if batch_idx < (n_batches - 1):
                s = slice(start_idx, start_idx + batchsize, 1)
                start_idx += batchsize

            else:
                s = slice(start_idx, len(table), 1)

            subset_table = table[s]
            g_idx = torch.from_numpy(subset_table.get_column("gene_idx").to_numpy().astype('int64'))
            s_idx = torch.from_numpy(subset_table.get_column("sample_idx").to_numpy().astype('int64'))
            score = torch.from_numpy(subset_table.get_column("score").to_numpy().astype('float32'))
            yield g_idx, s_idx, score
