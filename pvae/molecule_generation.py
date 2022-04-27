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
Args = Params()

print('############### data loading ######################')
data_dir = 'data/'
data_list = ['train_zinc','valid_zinc','test_zinc']

data_train,data_valid,data_test = text2dict_zinc(data_dir, data_list)
#txt, data_train_ = smi_postprocessing(data_train,biology_flag=False,LB_smi=15,UB_smi=120)

print("Number of SMILES >>>> " , len(data_train['SMILES']))



# smiles preprocessing:
'''
We filter the SMILES based on two criterias:
1) 8 < logP < -3 and 600 < MolecularWeight < 50,
2) 120 < smiles with length <  20

 pre-processing step is happening in smi_postprocessing is function.
 
 Please note that, biology_flag is served for the situations that the user has an access to biological label.
####
'''
biology_flag = False
LogP = [-3,8]
MW = [50,600]
LB_smi= 20
UB_smi = 120
txt,data_train_ = smi_postprocessing(data_train,biology_flag,LB_smi,UB_smi,LogP,MW)





'''
Dictionary building part: 
In this part, we generate dictionary based on the output of the smi_postrocessing function.

Then, we shape the data to the usable format for the pytorch using dataset_building function,
followed by DataLoader function.

'''

_,data_valid_ = smi_postprocessing(data_valid,biology_flag,LB_smi,UB_smi,LogP,MW)
data_type = 'pure_smiles'
char2ind,ind2char,sos_indx,eos_indx,pad_indx = dictionary_build(txt)
max_sequence_length = np.max(np.asarray([len(smi) for smi in data_train_['SMILES']]))

# here we create the dataset which we would like to sample from to generate novel molecules
# as an example we use samples here from the validation set.
Args.batch_size = 1 # as we need to work on a single sample in this case we need to set batch size to 1. 
dataset_valid = dataset_building(char2ind,data_valid_,max_sequence_length,data_type)
dataloader_valid = DataLoader(dataset=dataset_valid, batch_size= Args.batch_size, shuffle = False)


'''
Model loading step: 

We defined the model with pvae fuunction. reader for more detail of structure of network is encoraged to visit pvae.py.

'''
vocab_size = len(char2ind) 
model = SentenceVAE(vocab_size, Args.embedding_size, Args.rnn_type, Args.hidden_size, Args.word_dropout, Args.latent_size,
                    sos_indx, eos_indx, pad_indx, max_sequence_length,Args.nr_classes,Args.device_id,
                    num_layers=1,bidirectional=False, gpu_exist = Args.gpu_exist)
model.load_state_dict(torch.load('results/zinc_pure_smiles/E70.pytorch', map_location = 'cpu'))
model.cuda(Args.device_id)
    

for iteration, data in enumerate(dataloader_valid):
    count = 0 
    
    logp, mu, logv, z,prediction = model(to_var(data['input'], Args.device_id,gpu_exist = Args.gpu_exist),
                                          to_var(data['length'],Args.device_id,gpu_exist = Args.gpu_exist))
    for i in range(10):
        Rand_sample = to_var(torch.normal(mean=torch.zeros([1,  Args.latent_size]), std=0.01*torch.ones([1,  Args.latent_size])),
                                                 Args.device_id,gpu_exist=True)
        z_new = z+Rand_sample
        reconst = model.inference(z_new.cuda(Args.device_id))
        reconst = [ind2char[i] for i in reconst]
        reconst = decode(''.join(reconst))[0:-1]
    
    if iteration > 0:
        break