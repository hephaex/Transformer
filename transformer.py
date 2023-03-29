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
               
