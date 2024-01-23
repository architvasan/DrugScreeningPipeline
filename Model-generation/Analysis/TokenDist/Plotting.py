import seaborn as sns
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import json
import time
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 16}

matplotlib.rc('font', **font)
cmap="RdBu_r"

penguins = pd.read_csv('token_dist.csv')[:15]
penguins['Tot_comp']=[0 for i in range(len(penguins))]
df_tot = pd.read_csv('token_dist_tot_comb.csv')
df_tot_vals = []
for tok in penguins['Token']:
    print(df_tot[df_tot['Token']==tok]['Weight'].tolist()[0])
    df_tot_vals.append(df_tot[df_tot['Token']==tok]['Weight'].tolist()[0])

penguins['Tot_comp']=df_tot_vals
penguins.to_csv('data_comp.csv',index=False)
penguins.plot.barh(x='Token', stacked=True,figsize=(10,20), color=['red', 'gray'])
#penguins.plot(kind='bar', x=["Weight", "Tot_comp"], y="Token")#, stacked=True)#, legend=False)
plt.xlabel('Weight')
plt.ylabel('Token')
plt.savefig('token_dist_comp.png', dpi=300, bbox_inches='tight')
plt.close()
