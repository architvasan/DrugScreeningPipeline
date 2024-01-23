import seaborn as sns
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import json
#from SmilesPE.tokenizer import *
from smiles_pair_encoders_functions import *
import time
import numpy as np
import matplotlib.pyplot as plt

penguins = pd.read_csv('token_dist.csv')[:30]
sns.barplot(penguins, x="Weight", y="Token")#, legend=False)
plt.savefig('token_dist.png', dpi=300, bbox_inches='tight')
