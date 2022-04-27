import sys
sys.path.append('/home/ani/projects/chemical_vae/')
sys.path.append('/home/ani/miniconda3/envs/chemvae/lib/python3.6/site-packages/rdkit/')
sys.path.append('/home/ani/miniconda3/pkgs/xorg-libxrender-0.9.10-h516909a_1002/lib/')

import pandas as pd
import random
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
# vae stuff
from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu
# import scientific py
import numpy as np
import pandas as pd
# rdkit stuff
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools
import argparse

from tqdm import tqdm
import ast
#import seaborn as sn
import os


def get_embedding(smiles, model):
    try:
        X_1 = model.smiles_to_hot(mu.canon_smiles(smiles.strip()), canonize_smiles=True)
    except Exception as e:
        print(e)
        return None
    if X_1.size == 0:
        print('0 length')
        return None
    z_1 = model.encode(X_1)
    return z_1[0]

def canon_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
    
    return smiles

def get_vae_embeddings(smiles_list, model, include_dots = False):
    embeddings = {}
    invalid_smiles = []
    
    for smi in tqdm(smiles_list):
        emb = get_embedding(smi, model)
        if not emb is None:
            embeddings[smi] = emb
        else:
            invalid_smiles.append(smi)
                
    return embeddings, invalid_smiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help = 'csv file containing unique molecules and fingerprints')
    parser.add_argument('--vae_dir', help = 'Path to VAE model')
    parser.add_argument('--output')
    args = parser.parse_args()

    data_df = pd.read_csv(args.input)
    #data_df['fingerprint'] = data_df['fingerprint'].apply(ast.literal_eval)

    print('Loading VAE model...')
    print(args.vae_dir)
    vae = VAEUtils(directory=args.vae_dir)

    embeddings, invalid_smiles = get_vae_embeddings(data_df.smiles.values, vae)

    print(len(embeddings), len(invalid_smiles))
    print('Invalid SMILES percent:', len(invalid_smiles)/data_df.shape[0])

    vae_emb = []
    for smi in tqdm(data_df.smiles.values):
        if smi in embeddings:
            vae_emb.append(list(embeddings[smi]))
        else:
            vae_emb.append(np.nan)
            
    data_df['vae_emb'] = vae_emb
    data_df = data_df[~data_df.vae_emb.isna()]

    #data_df['fing_emb'] = [i + j for i, j in zip(data_df.fingerprint.values, data_df.vae_emb.values)]

    data_df = data_df.reset_index(drop = True)
    print('Dataset size after getting VAE embeddings:', data_df.shape)

    output_name = args.output.split('.csv')[0]  + '_' + str(data_df.shape[0]) + '.csv'
    print(f'Saving to {output_name}')
    data_df.to_csv(output_name)
