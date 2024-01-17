import torch
import numpy as np
import polars as pl
import pandas as pd
from sklearn.decomposition import PCA
from torch.utils.data import Dataset



# this class is for data loading of the normal variational autoencoder
class DepMap_Data(Dataset): 
    def __init__(self,
                 path):
        self.ds_path = path
        dep_map_data = pl.read_csv(path, separator = "\t")

        data = dep_map_data.melt(id_vars=['gene_id'],
                        variable_name='cell_line',
                        value_name='CERES_score',
                        )
        mean = data.select(pl.mean("CERES_score"))[0,0]
        sd = data.select(pl.std("CERES_score"))[0,0]

        dep_map_data = dep_map_data.with_columns((pl.col(dep_map_data.columns[1:]) - mean)/sd)

        # shuffle and split in train / test 
        dep_map_data = dep_map_data.sample(fraction = 1, shuffle = True)
        
        self.dataset = dep_map_data
        self.train_table = dep_map_data[:int(len(dep_map_data) * 0.9)]
        self.test_table = dep_map_data[int(len(dep_map_data) * 0.9):]

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
    

  

# this class is for data loading of the model suggested by the paper
class FasterDepMapDataset(Dataset):
    def __init__(self,
                 path,
                 gene_index_path):
        
        self.ds_path = path
        
        self.gene_index = pl.read_csv(gene_index_path)
        
        dep_map_data = pl.read_csv(path, separator = "\t")
        
        dep_map_data = dep_map_data.melt(id_vars=['gene_id'],
                                         variable_name='cell_line',
                                         value_name='CERES_score',
                                         )
        
        dep_map_data = dep_map_data.with_columns(pl.col('cell_line').cast(pl.Categorical))

        self.cell_line_index = dep_map_data.select(pl.col('cell_line').unique())
        self.cell_line_index = self.cell_line_index.with_row_count(name = 'cell_line_idx')
        
        dep_map_data = dep_map_data.join(self.gene_index, on='gene_id')
        dep_map_data = dep_map_data.join(self.cell_line_index, on='cell_line')
        dep_map_data = dep_map_data.with_columns(pl.col("cell_line_idx").cast(pl.Int64))
        
        # normalize by deviding by global std
        dep_map_data = dep_map_data.with_columns(pl.col('CERES_score') / pl.col('CERES_score').std())
        
        # shuffle and split in train / test 
        dep_map_data = dep_map_data.sample(fraction = 1, shuffle = True)
        
        self.dataset = dep_map_data
        self.train_table = dep_map_data[:int(len(dep_map_data) * 0.9)]
        self.test_table = dep_map_data[int(len(dep_map_data) * 0.9):]
        
                
    def compute_sample_init_pca(self, dim = 32, return_loadings = False, scale_pcs = True):
        
        dep_map_data = pl.read_csv(self.ds_path, separator = '\t')
        pca = PCA(dim)
        
        ds = dep_map_data.select(self.cell_line_index.get_column('cell_line').to_list()).to_numpy().transpose()
        ds = ds / ds.std()
        
        pcs = pca.fit_transform(ds)
        
        if scale_pcs is True:
            pcs = pcs / pcs.std(axis = 0)
        
        if return_loadings is True:
            return pcs, pca.components_.transpose(), dep_map_data.join(self.gene_index, on = 'gene_id', how='left').gene_idx.to_numpy()
        else: 
            return pcs
    
    
    def genes_in_assay(self):
        return self.train_table.get_column("gene_id").unique()
    
    
    def get_num_batches_per_epoch(self, batchsize):
        return int(np.ceil(len(self.train_table) / batchsize))
    
        
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
            g_idx = torch.from_numpy(subset_table.get_column("gene_idx").to_numpy().astype('int64'))
            s_idx = torch.from_numpy(subset_table.get_column("cell_line_idx").to_numpy().astype('int64'))
            score = torch.from_numpy(subset_table.get_column("CERES_score").to_numpy().astype('float32'))
            yield g_idx, s_idx, score
    
  