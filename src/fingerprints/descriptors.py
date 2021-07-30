import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
import ast
import argparse

import rdkit.Chem.Descriptors as descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFreeSASA
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Lipinski import NumHAcceptors
from rdkit.Chem.Lipinski import NumHDonors

def calculate_sasa(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        hmol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(hmol)
        radii = rdFreeSASA.classifyAtoms(hmol)
        return rdFreeSASA.CalcSASA(hmol, radii)
    except Exception as e:
        print(e)
        return np.nan 

def calculate_MW(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return descriptors.ExactMolWt(mol)
    except Exception as e:
        print(e)
        return np.nan

def calculate_logP(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return descriptors.MolLogP(mol)
    except Exception as e:
        print(e)
        return np.nan
    
def calculate_tpsa(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return CalcTPSA(mol)
    except Exception as e:
        print(e)
        return np.nan
    
    
def calculate_nHBAcc(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return NumHAcceptors(mol)
    except Exception as e:
        print(e)
        return np.nan
    
def calculate_nHBDon(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return NumHDonors(mol)
    except Exception as e:
        print(e)
        return np.nan


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help = 'csv file containing unique molecules and fingerprints')
    parser.add_argument('--descriptors', nargs="+", help = 'list of properties to be calculated and cancatenated to fingerprints')
    parser.add_argument('--output')
    args = parser.parse_args()

    desc_functions_dict = {'logP': calculate_logP, 'SASA': calculate_sasa, 'MW': calculate_MW, 
                      'TPSA': calculate_tpsa, 'nHBAcc': calculate_nHBAcc, 'nHBDon': calculate_nHBDon}

    data_df = pd.read_csv(args.input)
    data_df['fingerprint'] = data_df['fingerprint'].apply(ast.literal_eval)
    tqdm.pandas()

    for prop in args.descriptors:
        print(f'Calculating {prop}...')
        try:
            desc_function = desc_functions_dict[prop]
        except KeyError:
            print('Invalid Descriptor!!!')
            exit()

        data_df[prop] = data_df.SMILES.progress_apply(desc_function)
        # Replace Nan values with descriptor's mean
        prop_arr = np.array(data_df[prop])
        prop_mean = np.mean(prop_arr[~np.isnan(prop_arr)])
        data_df = data_df.fillna(prop_mean)


    scaled = preprocessing.StandardScaler().fit_transform(np.array(data_df[args.descriptors].values))
    for i, prop in enumerate(args.descriptors):
        data_df[prop + '_scaled'] = list(scaled[:, i])


    data_df['fing_desc'] = [f + list(d) for f, d in zip(data_df['fingerprint'].values, scaled)]
    print(len(data_df['fing_desc'].iloc[0]))

    output_name = args.output.split('.csv')[0]  + '_' + str(data_df.shape[0]) + '.csv'
    data_df.to_csv(output_name, index = False)

