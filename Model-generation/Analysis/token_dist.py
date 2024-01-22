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

##### Set up tokenizer ########
vocab_file = 'vocab_spe.txt'
spe_file = 'SPE_ChEMBL.txt'
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
print(dir(tokenizer))
df = pd.read_csv('../Sorted_redocked.csv')
for smi in df['SMILES']:
    print(smi)
    print(tokenizer.tokenize(smi))

