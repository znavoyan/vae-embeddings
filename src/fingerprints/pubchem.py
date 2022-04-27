import pandas as pd
import numpy as np
import re
from padelpy import from_smiles
from tqdm import tqdm
import argparse

def get_fingerprint(smiles):
    try:
        fingerprint = from_smiles(smiles, fingerprints=True, descriptors=False)
        return list(map(int, list(dict(fingerprint).values())))
    except:
        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help = 'Csv file containing unique molecules with SMILES')
    parser.add_argument('--output', help = 'Path to the file for saving molecules with fingerprints')
    args = parser.parse_args()

    data_df = pd.read_csv(args.input)
    print('Initial dataframe shape: ', data_df.shape)

    tqdm.pandas()
	data_df['fingerprint'] = data_df.SMILES.progress_apply(get_fingerprint)

	data_df = data_df[data_df['fingerprint'].apply(lambda x: len(x) != 0)]
	print('Final dataframe shape', data_df.shape)

	output_name = args.output.split('.csv')[0]  + '_' + str(data_df.shape[0]) + '.csv'
	print(f'Saving result to {output_name}')
	data_df.to_csv(output_name, index = False)