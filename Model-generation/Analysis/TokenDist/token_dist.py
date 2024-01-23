############# Module Loading ##############
from collections import OrderedDict
import csv
import argparse
import os
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import json
#from SmilesPE.tokenizer import *
from smiles_pair_encoders_functions import *
import time
import numpy as np

token_dist = {}

##### Set up tokenizer ########
vocab_file = 'vocab_spe.txt'
spe_file = 'SPE_ChEMBL.txt'
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
#print(dir(tokenizer))
df = pd.read_csv('../../Sorted_redocked.csv')
energy_dist = df['Scores']
energy_dist_mean = np.mean(df['Scores'])
energy_dist_std = np.std(df['Scores'])
#energy_part = np.sum([np.exp(-(e - energy_dist_mean)/float(energy_dist_std)) for e in energy_dist])
energy_part = np.sum([1 for e in energy_dist])
print(energy_part)

for smi,e in zip(df['SMILES'], df['Scores']):
    tokens_smi = tokenizer.tokenize(smi)
    for tok in tokens_smi:
        if not tok in token_dist:
            token_dist[tok]=1/float(energy_part)#np.exp(-(e - energy_dist_mean)/float(energy_dist_std))/float(energy_part)
        else:
            token_dist[tok]+=1/float(energy_part)#np.exp(-(e - energy_dist_mean)/float(energy_dist_std))/float(energy_part)
print(token_dist)
df_tok = pd.DataFrame(columns=['Token', 'Weight'])
df_tok['Token']=list(token_dist.keys())
df_tok['Weight']=list(token_dist.values())
#token_df = pd.DataFrame.from_dict([token_dist]).T
df_tok = df_tok.sort_values(by='Weight', ascending=False)
df_tok.to_csv('token_dist_unweighted.csv', index=False)

