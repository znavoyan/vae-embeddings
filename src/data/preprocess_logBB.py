import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit
print(rdkit.__version__)

import argparse
import sys
sys.path.append('/home/ani/projects/final_project/src')

from utils import percent_diff, canon_smiles

parser = argparse.ArgumentParser()
parser.add_argument('--output', help = 'Path to output file')
args = parser.parse_args()


data1_df = pd.read_csv('../../data/logBB/raw/y_test_indices.csv')
data2_sh1 = pd.read_excel('../../data/logBB/raw/LogBB dataset-new.xlsx', sheet_name = 'Experimental', engine = 'openpyxl')
data3_sh1 = pd.read_excel('../../data/logBB/raw/New log BB database.xlsx', sheet_name = 'LI, 415 compound', engine = 'openpyxl')
data3_sh2 = pd.read_excel('../../data/logBB/raw/New log BB database.xlsx', sheet_name = 'Abraham', 
                          engine = 'openpyxl').dropna(subset = ['SMILES'])
data3_sh3 = pd.read_excel('../../data/logBB/raw/New log BB database.xlsx', sheet_name = 'Subramanian', engine = 'openpyxl')


print(data1_df.shape, data2_sh1.shape, data3_sh1.shape, data3_sh2.shape, data3_sh3.shape)

data1_df['dataset'] = 'data1'

data2_sh1['BBclass'] = np.nan
data2_sh1['dataset'] = 'data2_sh1'

data3_sh1['logBB'] = np.nan
data3_sh1['BBclass'] = [1 if i == 'p' else 0 for i in data3_sh1.Class.values]
data3_sh1['dataset'] = 'data3_sh1'
data3_sh1 = data3_sh1[['SMILES', 'logBB', 'BBclass', 'dataset']]

data3_sh2 = data3_sh2[['SMILES', 'logBB']]

data3_sh2['BBclass'] = np.nan
data3_sh2['dataset'] = 'data3_sh2'

data3_sh3['logBB'] = np.nan
data3_sh3['BBclass'] = [1 if i == 'P' else 0 for i in data3_sh3['BBB+/BBB-'].values]
data3_sh3['dataset'] = 'data3_sh3'
data3_sh3 = data3_sh3[['SMILES', 'logBB', 'BBclass', 'dataset']]

data_df = pd.concat([data1_df, data2_sh1, data3_sh1, data3_sh2, data3_sh3], axis = 0)

data_df.SMILES = data_df.SMILES.apply(lambda x: x.strip())
print('All datasets:', data_df.shape)


data_df = data_df[~data_df.SMILES.apply(lambda x: '.' in x)]
print('After removing SMILES with dots:', data_df.shape)


# Remove Stereochemistry from SMILES and calculate InChIKey
can_smi_list = []
inchikeys = []

for s in data_df.SMILES.values:
    try:
        can_smi = canon_smiles(s.strip())
        ikey = Chem.MolToInchiKey(Chem.MolFromSmiles(can_smi))
    except Exception as e:
        #print(e)
        can_smi = ''
        ikey = ''
        
    can_smi_list.append(can_smi)
    inchikeys.append(ikey)
    
data_df['canon_smiles'] = can_smi_list
data_df['InchiKey'] = inchikeys

data_df = data_df[data_df.InchiKey != '']
print('Calculating InchiKey and removing invalid SMILES:', data_df.shape)

data_df = data_df.drop_duplicates(['InchiKey', 'logBB', 'BBclass']).reset_index(drop = True)
print('Unique InchiKeys:', len(set(data_df.InchiKey)))


original_smiles_list = []
inchikey_list = []
logBB_list = []
BBclass_list = []

for ikey, idx in data_df.groupby('InchiKey').groups.items():
    inchikey_list.append(ikey)
    group = data_df.iloc[idx]
    original_smiles_list.append(group.SMILES.values)
    logBB_list.append(group.logBB.values)
    BBclass_list.append(group.BBclass.values)
    
    
duplicated_df = pd.DataFrame({'InchiKey': inchikey_list, 'original_smiles': original_smiles_list, 
                           'logBB': logBB_list, 'BBclass': BBclass_list})


diff_percent_list = []
new_logbb_list = []
new_bbclass_list = []
signiff_diff_logbb_smiles = []

for i, row in duplicated_df.iterrows():
    logbbs = row.logBB
    bbclasses = row.BBclass
    
    # removing Nan values of logBB and BBclass
    logbbs = list(set(logbbs[~np.isnan(logbbs)]))
    bbclasses = list(set(bbclasses[~np.isnan(bbclasses)]))
    
    if len(logbbs) == 0: # no logBB values available, but we have classes
        diff_percent_list.append(np.nan)
        new_logbb_list.append(np.nan)
        
        if len(set(bbclasses)) == 2: # different classes, drop this molecule
            new_bbclass_list.append(np.nan)
        else:
            new_bbclass_list.append(bbclasses[0])
        
    elif len(logbbs) == 1: # there is one logBB value for this molecule 
        diff_percent_list.append(0.)
        new_logbb_list.append(logbbs[0])
        
        if logbbs[0] >= -1:
            new_bbclass_list.append(1)
        else:
            new_bbclass_list.append(0)
    else:
        temp = []
        for i in range(len(logbbs)-1):
            for j in range(i+1, len(logbbs)):
                temp.append(np.round(percent_diff(logbbs[i], logbbs[j]), 2))
                
        if np.mean(list(map(abs, temp))) <= 50: # logBB values are NOT signifficantly different
            new_logBB = np.mean(logbbs)
            new_logbb_list.append(new_logBB)
            if new_logBB >= -1:
                new_bbclass_list.append(1)
            else:
                new_bbclass_list.append(0)
            
        else: # logBB values are signifficantly different
            if np.all(np.array(logbbs) >= -1) or np.all(np.array(logbbs) < -1):
                new_logBB = np.mean(logbbs)
                new_logbb_list.append(new_logBB)
                if new_logBB >= -1:
                    new_bbclass_list.append(1)
                else:
                    new_bbclass_list.append(0)
            else:
                new_logbb_list.append(np.nan)
                new_bbclass_list.append(np.nan)
            
        diff_percent_list.append(np.mean(list(map(abs, temp))))
        
duplicated_df['diff_percent'] = diff_percent_list
duplicated_df['new_logBB'] = new_logbb_list
duplicated_df['new_BBclass'] = new_bbclass_list


duplicated_df['SMILES'] = [i[0] for i in duplicated_df.original_smiles.values]
duplicated_df = duplicated_df[~duplicated_df.new_BBclass.isna()].reset_index(drop = True)
print('Final dataset size:', duplicated_df.shape)

output_name = args.output.split('.csv')[0]  + '_' + str(duplicated_df.shape[0]) + '.csv'
print(f'Saving to {output_name}')
duplicated_df[['InchiKey', 'SMILES', 'new_logBB', 'new_BBclass']].to_csv(output_name, index = False)