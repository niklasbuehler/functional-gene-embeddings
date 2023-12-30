import scanpy as sc
import numpy as np
import polars as pl
import pandas as pd
import umap
import torch
import scipy.stats

from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, random_split
import anndata

device = torch.device("mps")
#torch.set_default_device(device) 


#dataset class for the compressed tabula sapiens dataset
class TS_Compressed_Dataset(Dataset):
    def __init__(self, 
                 gene_index,
                 path = '',
                 ):
        
        self.ds_path = path
        
        ts_data = sc.read_h5ad(path)
        
        #processing of the dataframe
        gene_list = ts_data.var.ensemblid.to_list()
        gene_list = [i.split(".")[0] for i in gene_list]
        ts_data.var_names = gene_list
        ts_data = ts_data.to_df()
        ts_data["sample_idx"] = ts_data.index
        ts_data.sample_idx = ts_data.sample_idx.apply(lambda x: int(x))
        
        #reshape the dataframe so that each row contains gene index, sample index and the score
        ts_data = ts_data.melt(id_vars=["sample_idx"], var_name="gene_id", value_name="score")
        ts_data = pd.merge(ts_data, gene_index, how = "inner", on = "gene_id")
        
        
        #normalize the score by the global mean and variance
        self.data_mean = ts_data.score.mean()
        self.data_std = ts_data.score.std()
        
        ts_data.score = (ts_data.score -  self.data_mean)/self.data_std
        
        ts_data = pl.from_pandas(ts_data)
        
        
        
        self.dataset = ts_data
        
        
        # shuffle and split in train / test 
        self.dataset = self.dataset.sample(fraction = 1, shuffle = True)
        
        self.train_table = self.dataset[:int(len(self.dataset) * 0.9)]
        self.test_table = self.dataset[int(len(self.dataset) * 0.9):]
        

    def compute_sample_init_pca(self, dim = 128):
        #compute the sample pcas for the initialization of sample embeddings
        ts_data = sc.read_h5ad(self.ds_path)
        ts_data = ts_data.to_df()
        ts_data = (ts_data - self.data_mean)/self.data_std
        
        pca = PCA(dim)
        
        pcs = pca.fit_transform(ts_data)
         
        return pcs
  

    def get_num_batches_per_epoch(self, batchsize):
        return int(np.ceil(len(self.train_table) / batchsize))
    

    def genes_in_assay(self):
        return self.train_table.get_column("gene_id").unique()
    
    
    def get_batches(self, batchsize, subset = 'train'):
        """
        this function returns a generator that gives the gene indices, sample indices and scores used in each batch
        """
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
            s_idx = torch.from_numpy(subset_table.get_column("sample_idx").to_numpy().astype('int64'))
            score = torch.from_numpy(subset_table.get_column("score").to_numpy().astype('float32'))
            yield g_idx, s_idx, score 
