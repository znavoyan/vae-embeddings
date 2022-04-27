import torch
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np
import pandas as pd

class dataset_building(Dataset):
# code is provided to shape the data with the format usable for 
# pytorch implementation 
# data_type: I divide the data in four categories:
# 1) physchem_biology 2) physchem 3) biology 4) pure_smiles
 
    def __init__(self,char2ind,data,max_seq_len,data_type):
        self.char2ind = char2ind
        self.data = data 
        self.max_seq_len = max_seq_len
        self.len = len(data['SMILES'])  
        self.smi2vec = self.vectorize_sequence()
        self.seq_len = self.get_len()
        self.data_type = data_type
        #print(self)
      
        
    def __getitem__(self,index):
        
        
        inputs= self.smi2vec[index][0:-1]
        targets = self.smi2vec[index][1:]
     
        seq_len = self.seq_len[index]-1
        #print(inputs, targets, seq_len)
        
        
        inputs_padd = Variable(torch.zeros((1, self.max_seq_len))).long()
        inputs_padd[0,:seq_len] = torch.LongTensor(inputs)
        target_padd = Variable(torch.zeros((1, self.max_seq_len))).long()
        target_padd[0,:seq_len] = torch.LongTensor(targets)
        
        if self.data_type == 'physchem_biology':
            
           
            effect  = self.data['Effect'][index]
            physchem  = self.data['physchem'][index]
            sample = {'input':    inputs_padd[0], 
                      'target':   target_padd[0],
                      'length':   seq_len,
                      'Effect':   effect,
                      'physchem': torch.FloatTensor(physchem)}
                      
        elif self.data_type == 'physchem':
            physchem  = self.data['physchem'][index]
            sample = {'input':    inputs_padd[0], 
                      'target':   target_padd[0],
                      'length':   seq_len,
                      'physchem': torch.FloatTensor(physchem)}
                      
        elif self.data_type == 'biology':
            effect  = self.data['Effect'][index]
            sample = {'input':    inputs_padd[0], 
                      'target':   target_padd[0],
                      'length':   seq_len,
                      'Effect':   effect}
                      
        elif self.data_type == 'pure_smiles':
        
            sample = {'input':    inputs_padd[0], 
                      'target':   target_padd[0],
                      'length':   seq_len}

        elif self.data_type == 'smiles_properties':
            prop = self.data['prop'][index]
            sample = {'input':    inputs_padd[0], 
                      'target':   target_padd[0],
                      'prop':     prop,
                      'length':   seq_len}

        else:
            print('request class does not exist!!!')
            #break 
        
        return sample
        
    def __len__(self):
        return self.len
    
    def vectorize_sequence(self):
        vectorized_seqs = [[self.char2ind[char] for char in smi]for smi in self.data['SMILES']]
        '''
                                vectorized_seqs = []
                                for smi in self.data['SMILES']:
                                    try:
                                        vectorized_seqs.append([self.char2ind[char] for char in smi])
                                    except Exception as e:
                                        print(e)
                                        print('Skipping ', smi)'''
       
        return vectorized_seqs
    
    def get_len(self):
        seq_len = [len(smi) for smi in self.data['SMILES']]
        return torch.LongTensor(seq_len)
    
    def padding(self):
        
        seqlen = len(self.seq)
        seq_tensor = Variable(torch.zeros((1, self.max_seq_len))).long()
        seq_tensor[:seqlen] = torch.LongTensor(self.seq)
        
        return  seq_tensor
            
            
            
def weightSampler(data):
    
    class_sample_count = np.array([len(np.where(data['Effect'] == t)[0]) for t in np.unique(data['Effect'] )])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in data['Effect']])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
    return sampler