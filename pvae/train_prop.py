from model_property2 import *
from data_prepration import *
from model_property2 import SentenceVAE
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
import argparse
import json
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--params_file',
                    help='parameters file', default='params.json')
parser.add_argument('-d', '--directory',
                    help='exp directory', default=None)
cmd_args = vars(parser.parse_args())
param_file = os.path.join(cmd_args['directory'], cmd_args['params_file'])
print(f'Reading parameters from {param_file}')
Args = json.loads(open(param_file).read(), object_pairs_hook=OrderedDict)

#Args = Params()

print('############### data loading ######################')
#data_dir = 'data/'
#data_list = ['zinc250k/zinc_train_logp_stand', 'zinc250k/zinc_test_logp_stand']

#if Args['data_list'][0][-3:] == 'csv':
data_train, data_valid = text2dict_zinc(Args['data_list'], Args['properties'])
#else:
#    data_train, data_valid = text2dict_zinc_txt(Args['data_dir'], Args['data_list'], Args['properties'])
# txt, data_train_ = smi_postprocessing(data_train,biology_flag=False,LB_smi=15,UB_smi=120)

print("Number of training SMILES >>>> ", len(data_train['SMILES']))
print("Number of validation SMILES >>>> ", len(data_valid['SMILES']))

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
#LogP = [-3, 8]
#MW = [50, 600]
#LB_smi = 20
UB_smi = 120
#txt, data_train_ = smi_postprocessing(data_train, biology_flag, LB_smi, UB_smi, LogP, MW)
#_, data_valid_ = smi_postprocessing(data_valid,biology_flag,LB_smi,UB_smi,LogP,MW)

txt, data_train_ = smi_postprocessing(data_train, biology_flag, UB_smi)
_, data_valid_ = smi_postprocessing(data_valid,biology_flag,UB_smi)
print(len(data_train_['SMILES']), len(data_train_['removed']))
print(len(data_valid_['SMILES']), len(data_valid_['removed']))

'''
Dictionary building part: 
In this part, we generate dictionary based on the output of the smi_postrocessing function.

Then, we shape the data to the usable format for the pytorch using dataset_building function,
followed by DataLoader function.

'''
data_type = 'smiles_properties'
char2ind, ind2char, sos_indx, eos_indx, pad_indx = dictionary_build(txt)
#max_sequence_length = np.max(np.asarray([len(smi) for smi in data_train_['SMILES']]))
max_sequence_length = UB_smi
dataset_train = dataset_building(char2ind, data_train_, max_sequence_length, data_type)
dataloader_train = DataLoader(dataset = dataset_train, batch_size = Args['batch_size'], shuffle = True)

dataset_valid = dataset_building(char2ind,data_valid_,max_sequence_length,data_type)
dataloader_valid = DataLoader(dataset = dataset_valid, batch_size = Args['batch_size'], shuffle = False)

print(len(dataset_train), len(dataset_valid))

'''
Model loading step: 

We defined the model with pvae fuunction. reader for more detail of structure of network is encoraged to visit pvae.py.

'''
vocab_size = len(char2ind)
'''
model = pvae(vocab_size, Args.embedding_size, Args.rnn_type, Args.hidden_size, Args.word_dropout, Args.latent_size,
                    sos_indx, eos_indx, pad_indx, max_sequence_length,Args.nr_classes,Args.device_id,
                    num_layers=1,bidirectional=False, gpu_exist = Args.gpu_exist)
'''
print('Latent_size:', Args['latent_size'])
model = SentenceVAE(vocab_size, Args['embedding_size'], Args['rnn_type'], Args['hidden_size'], Args['word_dropout'],
                    Args['latent_size'],
                    sos_indx, eos_indx, pad_indx, max_sequence_length, Args['device_id'],
                    num_layers=1, predict_prop = True, nr_prop = Args['nr_prop'], bidirectional=False, gpu_exist=Args['gpu_exist'])
if torch.cuda.is_available():
    torch.cuda.set_device(Args['device_id'])
    model = model.cuda(Args['device_id'])

class_weight = char_weight(txt, char2ind)
class_weight = torch.FloatTensor(class_weight).cuda(device=Args['device_id'])

optimizer = torch.optim.Adam(model.parameters(), lr=Args['learning_rate'])
NLL = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='sum', ignore_index=pad_indx)
#MSE = torch.nn.MSELoss()

step = 0
from loss_vae import *

tracker = defaultdict(list)
for epoch in range(0, Args['epochs']):

    optimizer, lr = adjust_learning_rate(optimizer, epoch, Args['learning_rate'])
    temp = defaultdict(list)
    score_acc = []
    AUC_acc = []

    #### Training
    for iteration, batch in enumerate(dataloader_train):
        batch = batch2tensor(batch, Args)
        batch_size = batch['input'].size(0)
        model.train()
        ######     Forward pass  ######
        logp, mean, logv, z, prediction = model(batch['input'], batch['length'])

        NLL_loss, KL_loss, KL_weight, prop_pred_loss = loss_fn(NLL, logp, batch['target'], batch['length'], mean, logv,
                                                               Args['anneal_function'], step, Args['k0'], Args['x0'], 
                                                               predict_prop = True, prop = batch['prop'], pred = prediction)
        loss = (NLL_loss + KL_weight * KL_loss + prop_pred_loss) / batch_size

        #### Evaluation #####
        temp['NLL_loss'].append(NLL_loss.item() / batch_size)
        temp['KL_loss'].append(KL_weight * KL_loss.item() / batch_size)
        temp['ELBO'].append(loss.item())
        temp['prop_pred_loss'].append(prop_pred_loss.item())

        #### Backward pass #####

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1  # for anneling

    #### Validation
    for iteration, batch in enumerate(dataloader_valid):
        batch = batch2tensor(batch, Args)
        batch_size = batch['input'].size(0) 
        
        logp, mean, logv, z, prediction = model(batch['input'], batch['length'])

        NLL_loss, KL_loss, KL_weight, prop_pred_loss = loss_fn(NLL, logp, batch['target'], batch['length'], mean, logv,
                                                               Args['anneal_function'], step, Args['k0'], Args['x0'], 
                                                               predict_prop = True, prop = batch['prop'], pred = prediction)
        loss = (NLL_loss + KL_weight * KL_loss + prop_pred_loss) / batch_size

        #### Evaluation #####
        temp['val_NLL_loss'].append(NLL_loss.item() / batch_size)
        temp['val_KL_loss'].append(KL_weight * KL_loss.item() / batch_size)
        temp['val_ELBO'].append(loss.item())
        temp['val_prop_pred_loss'].append(prop_pred_loss.item())
    
    # for iteration, batch in enumerate(dataloader_test):
    #    batch = batch2tensor(batch,Args)
    #   batch_size = batch['input'].size(0)
    #  model.train()
    ######     Forward pass  ######
    # logp, mean, logv, z,prediction = model(batch['input'], batch['length'])
    # model.eval()

    tracker['NLL_loss'].append(np.mean(np.asarray(temp['NLL_loss'])))
    tracker['KL_loss'].append(np.mean(np.asarray(temp['KL_loss'])))
    tracker['ELBO'].append(np.mean(np.asarray(temp['ELBO'])))
    tracker['prop_pred_loss'].append(np.mean(np.asarray(temp['prop_pred_loss'])))

    tracker['val_NLL_loss'].append(np.mean(np.asarray(temp['val_NLL_loss'])))
    tracker['val_KL_loss'].append(np.mean(np.asarray(temp['val_KL_loss'])))
    tracker['val_ELBO'].append(np.mean(np.asarray(temp['val_ELBO'])))
    tracker['val_prop_pred_loss'].append(np.mean(np.asarray(temp['val_prop_pred_loss'])))

    print(" epoch %04d/%i, 'ELBO' %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, prop_pred_loss %9.4f, KL-Weight %6.3f, lr %9.7f"
          % (epoch, Args['epochs'],
             np.mean(np.asarray(tracker['ELBO'])),
             np.mean(np.asarray(tracker['NLL_loss'])),
             np.mean(np.asarray(tracker['KL_loss'])),
             np.mean(np.asarray(tracker['prop_pred_loss'])),
             KL_weight,
             lr))

    print("\t\t  'val_ELBO' %9.4f, val_NLL-Loss %9.4f, val_KL-Loss %9.4f, val_prop_pred_loss %9.4f"
          % (np.mean(np.asarray(tracker['val_ELBO'])),
             np.mean(np.asarray(tracker['val_NLL_loss'])),
             np.mean(np.asarray(tracker['val_KL_loss'])),
             np.mean(np.asarray(tracker['val_prop_pred_loss']))))

    if epoch % Args['save_every'] == 0:
        checkpoint_path = os.path.join(cmd_args['directory'], "E%i.pytorch" % (epoch))
        torch.save(model.state_dict(), checkpoint_path)
save_loss_path = os.path.join(cmd_args['directory'], "tracker.pickle")
with open(save_loss_path, 'wb') as f:
    pickle.dump(tracker, f)
