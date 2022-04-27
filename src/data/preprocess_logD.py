import numpy as np
import pandas as pd
from rdkit import Chem
import argparse

import sys
sys.path.append('/home/ani/projects/vae-fingerprints/src')

from utils import percent_diff, canon_smiles


parser = argparse.ArgumentParser()
parser.add_argument('--output', help = 'Path to the output file')
args = parser.parse_args()


data_df = pd.read_excel('/home/ani/projects/vae-fingerprints/data/logD/raw/logD_shake flask.xlsx', engine = 'openpyxl')
print('Initial dataset size:', data_df.shape[0])

data_df = data_df[(data_df['unknown2'] != '>') & (data_df['unknown2'] != '<')].dropna(subset = ['logD'])

data_df = data_df[~data_df.SMILES.apply(lambda x: '.' in x)]
print('After removing SMILES with dots:', data_df.shape[0])

data_df = data_df[(data_df.logD <= 6) & (data_df.logD >= -2)]
print(data_df.shape)

# Canonize SMILES, removing sterechemistry and construct InChIKeys
can_smi_list = []
inchikeys = []

for s in data_df.SMILES.values:
    try:
        can_smi = canon_smiles(s.strip())
        ikey = Chem.MolToInchiKey(Chem.MolFromSmiles(can_smi))
    except:
        print(s)
        can_smi = ''
        ikey = ''
        
    can_smi_list.append(can_smi)
    inchikeys.append(ikey)
    
data_df['canon_smiles'] = can_smi_list
data_df['InchiKey'] = inchikeys

data_df = data_df[data_df.InchiKey != ''].reset_index(drop = True)
print('Number of unique InChIKeys:', len(set(data_df.InchiKey.values)))

# build dataframe with duplicated entries
original_smiles_list = []
inchikey_list = []
logD_list = []

for ikey, idx in data_df.groupby('InchiKey').groups.items():
    inchikey_list.append(ikey)
    group = data_df.iloc[idx]
    original_smiles_list.append(group.SMILES.values)
    logD_list.append(group.logD.values)
    
duplicated_df = pd.DataFrame({'InchiKey': inchikey_list, 'original_smiles': original_smiles_list, 
                           'logD': logD_list})

print('Number of molecules with multiple labels:', duplicated_df[duplicated_df.logD.apply(lambda x: len(x) > 1)].shape[0])

diff_percent_list = []
new_logD_list = []

for i, row in duplicated_df.iterrows():
    logd_values = row.logD
        
    if len(logd_values) == 1:
        diff_percent_list.append(0.)
        new_logD_list.append(logd_values[0])
        
    else:
        diff_perc_temp = []
        
        for i in range(len(logd_values)-1):
            for j in range(i+1, len(logd_values)):
                diff_perc_temp.append(np.round(percent_diff(logd_values[i], logd_values[j]), 2))
                
        # logS values are NOT signifficantly different
        if np.mean(list(map(abs, diff_perc_temp))) <= 50: 
            new_logD_list.append(np.mean(logd_values))
            
        else: # logS values are signifficantly different
            new_logD_list.append(np.nan)
            
        diff_percent_list.append(np.mean(list(map(abs, diff_perc_temp))))
        
duplicated_df['diff_percent'] = diff_percent_list
duplicated_df['new_logD'] = new_logD_list
duplicated_df['SMILES'] = [i[0] for i in duplicated_df.original_smiles.values]
print(duplicated_df.shape)

duplicated_df = duplicated_df.dropna(subset = ['new_logD'])
print('Final dataset size:', duplicated_df.shape[0])

duplicated_df = duplicated_df[['InchiKey', 'SMILES', 'logD', 'new_logD']]

# Save the Data
output_name = args.output.split('.csv')[0]  + '_' + str(duplicated_df.shape[0]) + '.csv'
print(f'Saving to {output_name}')
duplicated_df.to_csv(output_name, index = False)