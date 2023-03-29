import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

fix_seed = 2023
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'                      
               
class Dataset(Dataset):
    def __init__(self, flag, seq_len, pred_len):
        #
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        #
        type_map = {'train': 0, 'val': 1, 'test':2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        
        #seaborn 
        df_raw = sns.load_dataset('flights')

        #
        border1s = [0, 12 * 9 - self.seq_len, 12 * 11 - self.seq_len]
        border2s = [12 * 9, 12 * 11, 12 * 12]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw[['passengers']].values
        ss = StandardScaler()
        data = ss.fit_transform(data)

        self.data = data[border1:border2]

    def __getitem__(self, index):
        #
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        src = self.data[s_begin:s_end]
        tgt = self.data[r_begin:r_end]

        return src, tgt
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
def data_provider(flag, seq_len, pred_len, batch_size):
    #flag
    data_set = Dataset(flag=flag, 
                       seq_len=seq_len, 
                       pred_len=pred_len
                       )
    #Data loader
    data_loader = DataLoader(data_set,
                             batch_size=batch_size, 
                             shuffle=True
                            )
    
    return data_loader
