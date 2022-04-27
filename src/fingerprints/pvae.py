import sys
sys.path.append('/home/ani/projects/pvae/')

from model_property2 import *
from data_prepration import *
from params import *
from multiprocessing import cpu_count
import numpy as np
import pickle
from util import *
import pickle 
import os
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import json
from collections import OrderedDict
from tqdm import tqdm
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help = 'csv file containing unique molecules and fingerprints')
    parser.add_argument('--model_dir', help = 'Path to PVAE model')
    parser.add_argument('--model_name', default = 'E70.pytorch', help = "Pretrained model's name")
    parser.add_argument('--output')
    args = parser.parse_args()

    param_file = os.path.join(args.model_dir, 'params.json')
    model_path = os.path.join(args.model_dir, args.model_name)

    print(f'Reading parameters from {param_file}')
    Args = json.loads(open(param_file).read(), object_pairs_hook=OrderedDict)
    train_path = Args['data_list'][0]
    data_train, data_valid = text2dict_zinc([train_path, args.input])
    print("Number of training SMILES >>>> " , len(data_train['SMILES']))
    print("Number of input SMILES >>>> " , len(data_valid['SMILES']))

    biology_flag = False
    UB_smi = 120
    print('postprocessing train...')
    txt,data_train_ = smi_postprocessing(data_train, biology_flag, UB_smi)
    print('dictionary building...')
    char2ind,ind2char,sos_indx,eos_indx,pad_indx = dictionary_build(txt)

    print('postprocessing input...')
    _,data_valid_ = smi_postprocessing(data_valid, biology_flag, UB_smi, char2ind)
    print(len(data_valid_['SMILES']))


    data_type = 'pure_smiles'
    max_sequence_length = UB_smi

    print('dataset building...')
    Args.batch_size = 1 # as we need to work on a single sample in this case we need to set batch size to 1. 
    dataset_valid = dataset_building(char2ind,data_valid_,max_sequence_length,data_type)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size= Args.batch_size, shuffle = False)


    print('Loading model: ', model_path)
    vocab_size = len(char2ind) 
    if 'predict_prop' in Args:
        print('Property predicting model: ', Args['predict_prop'])
        model = SentenceVAE(vocab_size, Args['embedding_size'], Args['rnn_type'], Args['hidden_size'], Args['word_dropout'], Args['latent_size'],
                            sos_indx, eos_indx, pad_indx, max_sequence_length, Args['device_id'],
                            num_layers=1, predict_prop = Args['predict_prop'], nr_prop = Args['nr_prop'], bidirectional=False, gpu_exist = Args['gpu_exist'])
        model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
        model.cuda(Args['device_id'])
    else:
        print('Property predicting model: False')
        model = SentenceVAE(vocab_size, Args['embedding_size'], Args['rnn_type'], Args['hidden_size'], Args['word_dropout'], Args['latent_size'],
                            sos_indx, eos_indx, pad_indx, max_sequence_length, Args['device_id'],
                            num_layers=1, nr_prop = Args['nr_classes'], bidirectional=False, gpu_exist = Args['gpu_exist'])
        model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
        model.cuda(Args['device_id'])


    print('Getting embeddings...')
    embeddings = {}
    for data, smiles in tqdm(zip(dataloader_valid, dataset_valid.data['SMILES_org'])):
        if 'predict_prop' in Args:
            _, _, _, z, _ = model(to_var(data['input'], Args['device_id'], gpu_exist = Args['gpu_exist']),
                                                  to_var(data['length'], Args['device_id'], gpu_exist = Args['gpu_exist']))
        else:
            _, _, _, z = model(to_var(data['input'], Args['device_id'], gpu_exist = Args['gpu_exist']),
                                                  to_var(data['length'], Args['device_id'], gpu_exist = Args['gpu_exist']))
        embeddings[smiles] = z.tolist()[0]


    input_data = pd.read_csv(args.input)
    input_data = input_data[input_data.smiles.apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)).strip() in embeddings.keys())]
    print(input_data.shape)

    input_data['vae_emb'] = [embeddings[Chem.MolToSmiles(Chem.MolFromSmiles(smi))] for smi in input_data.smiles.values]
    print(len(input_data['vae_emb'][0]))

    output_name = args.output.split('.csv')[0]  + '_' + str(input_data.shape[0]) + '.csv'
    print(f'Saving to {output_name}')
    input_data.to_csv(output_name)