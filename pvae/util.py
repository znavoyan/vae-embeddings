import numpy as np
#from tqdm import *
import itertools
from collections import defaultdict, Counter
import rdkit 
from rdkit import Chem
from rdkit.Chem import Descriptors
#import model_attention
import torch 
import torch.nn as nn
from sklearn.metrics import roc_auc_score, r2_score
import pandas as pd
import os


def text2dict_zinc_txt(data_dir, data_list, predict_prop = False):
    # This function is used to process the input data
    # taken from ZINC dataset
    #### OUTPUT #####
    # Dictionary of training, testing and validation data
    ## Sadegh Mohammadi, BCS, Monheim, 07.Nov.2018
    data_train = defaultdict(list)
    data_test = defaultdict(list)
    #data_valid = defaultdict(list)
    
    for data in data_list:
        print(data_dir+ data+'.txt')       
        with open(data_dir+ data+'.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                smi = line[0].strip()
        
                try:
                    prop = np.double(line[1].strip())
                except:
                    prop = np.nan
                #mw = np.double(line[2].strip())
                #tpsa = np.double(line[1].strip())

                if data == data_list[0]:
                    data_train['SMILES'].append(smi)
                    data_train['prop'].append(prop)
                    #data_train['MW'].append(mw)
                    #data_train['tpsa'].append(tpsa)
                if data == data_list[1]:
                    data_test['SMILES'].append(smi)
                    data_test['prop'].append(prop)
                    #data_test['MW'].append(mw)
                    #data_test['tpsa'].append(tpsa)
                    
                #if data == data_list[2]:
                #    data_valid['SMILES'].append(smi)
                #    data_valid['logP'].append(logp)
                #    data_valid['MW'].append(mw)
    return data_train, data_test


def text2dict_zinc(data_list, properties = None):
    # This function is used to process the input data
    # taken from ZINC dataset
    #### OUTPUT #####
    # Dictionary of training, testing and validation data
    ## Sadegh Mohammadi, BCS, Monheim, 07.Nov.2018
    data_train = defaultdict(list)
    data_test = defaultdict(list)

    #print(os.path.join(data_dir, data_list[0]))
    #print(os.path.join(data_dir, data_list[1]))
    #train_df = pd.read_csv(os.path.join(data_dir, data_list[0]))
    #test_df = pd.read_csv(os.path.join(data_dir, data_list[1]))
    print(data_list[0])
    print(data_list[1])
    train_df = pd.read_csv(data_list[0])
    test_df = pd.read_csv(data_list[1])
    print('Train:', train_df.shape[0])
    print('Test:', test_df.shape[0])
    
    #print(train_df.columns)
    
    data_train['SMILES'] = train_df['smiles'].values
    data_test['SMILES'] = test_df['smiles'].values

    if properties:
        print('Properties:', properties)
        data_train['prop'] = np.array(list(train_df[properties].values))
        data_test['prop'] = np.array(list(test_df[properties].values))
    else:
        data_train['prop'] = np.empty((len(data_train['SMILES']),))
        data_train['prop'].fill(np.nan)
        data_test['prop'] = np.empty((len(data_test['SMILES']),))
        data_test['prop'].fill(np.nan)

    return data_train, data_test

def smi_postprocessing(data,biology_flag,UB_smi, char2ind = None):
    #print(data['SMILES'])
    from collections import defaultdict
    
    saver = defaultdict(list)
    data_postproces = {} 
    text = ""
    #okchars = "abefg-+ACFLRIONPScons12345678=#()" # .M,9
    okchars = "CFONSPLRAIYVXQrvabefghxywzmklqicons.12345678=#()-"
   
    text = ""
    smi_list = []
    #for indx, smi, tpsa in zip(np.arange(len(data['SMILES'])), data['SMILES'], data['tpsa']):
    for indx, smi, prop in zip(np.arange(len(data['SMILES'])), data['SMILES'], data['prop']):
        #logp = Descriptors.MolLogP(Chem.MolFromSmiles(smi))
        #mw = Descriptors.MolWt(Chem.MolFromSmiles(smi))
        # decreasing number of characters in the training set
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi.strip()))
        smir = smi.replace("Cl","L")
        smir = smir.replace("Br","R")
        
        smir = smir.replace("[nH]","A")
        smir = smir.replace('[nH+]','r')
        smir = smir.replace('[NH+]','a')
        smir = smir.replace('[NH2+]','b')
        smir = smir.replace('[NH3+]','e')
        smir = smir.replace("[N+]","f")
        
        smir = smir.replace("[O-]","g") 
        smir = smir.replace("[Si]","h")  
        
        smir = smir.replace('[N-]','v')
        smir = smir.replace('[n+]','V')
        smir = smir.replace('[n-]','w')
        smir = smir.replace('[S-]','s')
        smir = smir.replace('[NH-]','l')
        smir = smir.replace('[o+]','q')
        
        # removes carbon stereo - optional
        smir = smir.replace("[C@]","x")
        smir = smir.replace("[C@@]","X")
        smir = smir.replace("[C@H]","y")
        smir = smir.replace("[C@@H]","Y") 
        
        smir = smir.replace("[S@@]","Q")
        
        smir = smir.replace("[S@]","z")
        
        smir = smir.replace("[P@@]","m")
        smir = smir.replace("[P@]","k")
        
        smir = smir.replace("[P@@H]","i")
       
        
        # removes bond EZ stereo
        smir = smir.replace("/","")
        smir = smir.replace("\\","")
        #print(smir)
        # removes SMILES with nonallowed characters
        ok = all(c in okchars for c in smir)
        if not ok: 
            saver['removed'].append(smir)
            continue
            #saver['no'].append(("G" + smi + "E"))  
            #saver['no'].append(indx)
        #if len(smi) > LB_smi and len(smi) < UB_smi and logp >=LogP[0] and logp <=LogP[1] and mw >= MW[0] and mw <= MW[1] : 
        if len(smi) <= UB_smi:    
            text +=  "G" + smir + "E"
            #print("yes")
            #smi_list.append(("G" + smi + "E"))
            smiles = "G" + smir + "E"
            if char2ind:
                try:
                    [char2ind[char] for char in smiles]
                    saver['SMILES'].append((smiles))
                    saver['SMILES_org'].append( smi )
                    saver['prop'].append(prop)
                    if biology_flag == True:
                        saver['Effect'].append(data['Effect'][indx])
                except Exception as e:
                    print(e)
                    print('Skipping ', smiles)
                    continue
            else:
                saver['SMILES'].append((smiles))
                saver['SMILES_org'].append( smi )
                saver['prop'].append(prop)
                if biology_flag == True:
                    saver['Effect'].append(data['Effect'][indx])
    #data_postproces={'SMILES':smi_list,'Effect':Effect}
    return text, saver

# INCORRECT
def decode(smi):
    
    smi = smi.replace("L","Cl")
    smi = smi.replace("R","Br")
    smi = smi.replace("A","[nH]")
    smi = smi.replace('r','[nH+]')
    smi = smi.replace('a','[NH+]')
    smi = smi.replace('b','[NH2+]')
    smi = smi.replace('e','[NH3+]')
    smi = smi.replace("f","[N+]")
    smi = smi.replace("g","[O-]")
    smi = smi.replace("h","[Si]")
    smi = smi.replace("x","[C@]")
    smi = smi.replace("X","[C@@]")
    smi = smi.replace("X","[C@@]")
    smi = smi.replace("y","[C@H]")
    smi = smi.replace("Y","[C@@H]")
    smi = smi.replace("Q","[S@@]")
    smi = smi.replace("z","[S@]")
    smi = smi.replace("m","[P@@]")
    smi = smi.replace("k","[P@]")
    smi = smi.replace("i","[P@@H]") 
    smi = smi.replace('v','[N-]')
    smi = smi.replace('V','[n+]')
    smi = smi.replace('w','[n-]')
    smi = smi.replace('s','[S-]')
    smi = smi.replace('l','[NH-]')
    smi = smi.replace('q','[o+]')
   
    return smi

# CORRECT
def decode_1(smi):
    
    smi = smi.replace('r','[nH+]')
    smi = smi.replace('l','[NH-]')
    smi = smi.replace("i","[P@@H]") 
    smi = smi.replace("L","Cl")
    smi = smi.replace("R","Br")
    smi = smi.replace("A","[nH]")
    smi = smi.replace('a','[NH+]')
    smi = smi.replace('b','[NH2+]')
    smi = smi.replace('e','[NH3+]')
    smi = smi.replace("f","[N+]")
    smi = smi.replace("g","[O-]")
    smi = smi.replace("h","[Si]")
    smi = smi.replace("x","[C@]")
    smi = smi.replace("X","[C@@]")
    smi = smi.replace("X","[C@@]")
    smi = smi.replace("y","[C@H]")
    smi = smi.replace("Y","[C@@H]")
    smi = smi.replace("Q","[S@@]")
    smi = smi.replace("z","[S@]")
    smi = smi.replace("m","[P@@]")
    smi = smi.replace("k","[P@]")
    smi = smi.replace('v','[N-]')
    smi = smi.replace('V','[n+]')
    smi = smi.replace('w','[n-]')
    smi = smi.replace('s','[S-]')
    smi = smi.replace('q','[o+]')
   
    return smi


def check_all_key_in_test(data, char2ind):
    data_ = {}
    acc = []
    okchars = list(char2ind.keys())
    for smi in data['SMILES']:
        ok = all(c in okchars for c in smi)
        if ok:
            acc.append(smi)
    data_['SMILES'] = acc
    return data_
    

##### Physchem properties

def physchem_extract(data,physname):
    # here we extract the physchem properties
    if physname == 'MWt':
        MWt = np.asarray([ Descriptors.MolWt(Chem.MolFromSmiles(decode(smi[1:-1]))) for smi in data['SMILES']])
        min_MWt = np.min(MWt)
        max_MWt = np.max(MWt)
        return min_MWt, max_MWt, MWt
    elif physname == 'LogP':
        MLogP = np.asarray([Descriptors.MolLogP(Chem.MolFromSmiles(decode(smi[1:-1]))) for smi in data['SMILES']])
        min_MLogP = np.min(MLogP)
        max_MLogP = np.max(MLogP)
        return min_MLogP, max_MLogP, MLogP
    elif physname == 'MWt_LogP':
        MWt = np.asarray([Descriptors.MolWt(Chem.MolFromSmiles(decode(smi[1:-1]))) for smi in data['SMILES']])
        MLogP = np.asarray([Descriptors.MolLogP(Chem.MolFromSmiles(decode(smi[1:-1]))) for smi in data['SMILES']])
        physchems = np.vstack((MWt, MLogP)).transpose(1,0)
        min_phys = np.min(physchems,axis = 0)
        max_phys = np.max(physchems, axis = 0)
        return min_phys, max_phys, physchems
    else:
        raise ValueError('Keyword ' % (physname),' does not exist')
def max_min_norm(Descriptors,min_tr, max_tr):
    
    Descriptors_norm = (Descriptors - min_tr)/(max_tr - min_tr)
    
    return Descriptors_norm
def max_min_rescale(Descriptors,min_tr,max_tr):
    rescaling = (Descriptors*(max_tr - min_tr)) + min_tr
    return rescaling
    
def max_norm(Descriptors, max_tr):
    
    Descriptors_norm = (Descriptors )/(max_tr)
    
    return Descriptors_norm
    

def physchem_normalized(physchems):
    # The input is physchem properties, Mwt, MolLogP, TPSA,
    # Mwt_tr,Mlp_tr,Tpsa_tr: Physchem properties of the training set
    ####### OUTPUT #####
    # Normalized physchems ( Mwt, MolLogP, TPSA)
    from sklearn import preprocessing
    scaler   = preprocessing.StandardScaler().fit(physchems)
    physchemsn = scaler.transform(physchems)  
    return scaler,physchemsn   

def dictionary_build(txt):
    # here we shape the dictionary given the raw smiles txt
    pad_indx = 0
    pad_char = ' '
    char2ind,ind2char=symdict(txt,1)
    ind2char.update({pad_indx: pad_char })
    char2ind.update({pad_char: pad_indx })
    sos_indx = char2ind['G']
    eos_indx = char2ind['E']
    return char2ind,ind2char,sos_indx,eos_indx,pad_indx

def symdict(txt,offset):
    # This function is used for converting chemical symbols
    # to index and reverse one.
    #chars = ['<pad>'] + sorted(list(set(txt)))
    chars = sorted(list(set(txt)))
    print('total chars:', len(chars))
    char_indices = dict((c, i+offset) for i, c in enumerate(chars))
    indices_char = dict((i+offset, c) for i, c in enumerate(chars))
    return char_indices,indices_char

def char_weight(txt,char2ind):
    count = Counter(txt).most_common(len(char2ind))

    coocurrance = list((dict(count).values()))
    symbols = list((dict(count).keys()))
    lamda_factor =  np.log(coocurrance)/np.sum(np.log(coocurrance))
    lamda_factor = (1/(lamda_factor+0.000001))*0.01
    weights = {}
    for i,element in enumerate(symbols):
        if lamda_factor[i] > 1.:
            lamda_factor[i] = 0.90
        weights[element] = lamda_factor[i]

    #print(weights)
    class_weight = []
    for element in char2ind:
        if element == ' ':
            weight = 0.003
        else: 
            weight = weights[element]
        class_weight.append(weight)
    return class_weight 


def kl_anneal_function(anneal_function, step, k0, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k0*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

    
def randint(data,n_samples,rseed):
    import random 
    random.seed(rseed)
    idx_list = []
    while len(idx_list) < n_samples:
        idx = random.randint(1,len(data)-1)
        if idx not in idx_list:
            idx_list.append(idx)
    return idx_list

#### Training relevent #######

def training(class_weight,dataloader_train,dataloader_test,pad_indx,model,opts):
    
    accumulator = defaultdict(list)
    step = 0 # 
    optimizer = torch.optim.Adam(model.parameters(), lr = opts.learning_rate)
    
    tracker = defaultdict(list)
    
    NLL = torch.nn.CrossEntropyLoss(weight = class_weight, reduction = 'sum', ignore_index = pad_indx)
   
    if opts.data_type == 'physchem':
        MSE = nn.MSELoss()
        
    elif opts.data_type == 'biology':
        BCE = nn.BCEWithLogitsLoss()
        
    elif opts.data_type == 'physchem_biology':
        MSE = nn.MSELoss()
        BCE = nn.BCEWithLogitsLoss()     
    elif opts.data_type == 'pure_smiles':
        print('only pure smiles')
    else:
        raise ValueError() 
    for epoch in range(0, opts.epochs):
        optimizer,lr = pvae.adjust_learning_rate(optimizer, epoch, opts.learning_rate)
        temp = defaultdict(list)
        print('epoch >>>> ', epoch )
        
       
        temp = defaultdict(list)
        for iteration, batch in enumerate(dataloader_test):
           # try:
                batch = pvae.batch2tensor(batch,opts)
                batch_size = batch['input'].size(0)
                model.train()
                  ######     Forward pass  ######
                logp, mean, logv, z,prediction = model(batch['input'], batch['length'])
                prediction = prediction.squeeze()
                NLL_loss, KL_loss, KL_weight = pvae.loss_fn(NLL,logp, batch['target'],batch['length'], mean, logv, opts.anneal_function, step, opts.k0,opts.x0)
                if opts.data_type == 'physchem':
                    phys_loss = MSE(prediction,batch['physchem'].squeeze())
                    loss = (NLL_loss + KL_weight * KL_loss + 100*phys_loss)/batch_size
                    temp['ELBO'].append(loss.item())
                    temp['NLL_loss'].append(NLL_loss.item()/batch_size)
                    temp['KL_loss'].append(KL_weight*KL_loss.item()/batch_size)
                    temp['phys_loss'].append(100*phys_loss.item()/batch_size)
                    temp['score_train'].append(r2_score(batch['physchem'].data.cpu().numpy(),prediction.squeeze().data.cpu().numpy()))
                         
                
                elif opts.data_type == 'biology':
                    bio_loss = BCE(prediction,batch['Effect'])
                    loss = (NLL_loss + KL_weight * KL_loss + 100*bio_loss)/batch_size
                    temp['ELBO'].append(loss.item())
                    temp['NLL_loss'].append(NLL_loss.item()/batch_size)
                    temp['KL_loss'].append(KL_weight*KL_loss.item()/batch_size)
                    temp['bio_loss'].append(100*bio_loss.item()/batch_size) 
                    temp['AUC_train'].append(roc_auc_score(batch['Effect'].data.cpu().numpy(),prediction.squeeze().data.cpu().numpy()))
        
                elif opts.data_type == 'physchem_biology':
                    print(prediction.shape, batch['physchem'].shape)
                    phys_loss = MSE(prediction[:,0:-1],batch['physchem'])
                    bio_loss  = BCE(prediction[:,-1],batch['Effect'])
                    loss = (NLL_loss + KL_weight * KL_loss + 100*phys_loss + 100*bio_loss)/batch_size
                    temp['ELBO'].append(loss.item())
                    temp['NLL_loss'].append(NLL_loss.item()/batch_size)
                    temp['KL_loss'].append(KL_weight*KL_loss.item()/batch_size)
                    temp['phys_loss'].append(100*phys_loss.item()/batch_size)
                    temp['bio_loss'].append(100*bio_loss.item()/batch_size) 
                    temp['score_train'].append(r2_score(batch['physchem'].data.cpu().numpy(),prediction[:,0:-1].squeeze().data.cpu().numpy()))
                    temp['AUC_train'].append(roc_auc_score(batch['Effect'].data.cpu().numpy(),prediction[:,-1].squeeze().data.cpu().numpy()))
                    
                elif opts.data_type == 'pure_smiles':
                    
                    loss = (NLL_loss + KL_weight * KL_loss)/batch_size
                    temp['ELBO'].append(loss.item())
                    temp['NLL_loss'].append(NLL_loss.item()/batch_size)
                    temp['KL_loss'].append(KL_weight*KL_loss.item()/batch_size)
                    
                else:
                    raise ValueError()
                    print(opts.data_type, 'does not exist')
                    
            #except:
             #   continue
             #   raise ValueError() 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
       
        temp_test = test(model,dataloader_test,opts)
        #print(temp, temp_test)
        tracker = loss_saver(accumulator,temp,temp_test,epoch,opts, lr,KL_weight)
    return tracker

def test(model,dataloader_test,opts):
     
    if opts.data_type != 'pure_smiles':
        temp = defaultdict(list)
        for iteration, batch in enumerate(dataloader_test):
                #try:
            batch = pvae.batch2tensor(batch,opts)
            batch_size = batch['input'].size(0)
            model.eval()
                      ######     Forward pass  ######
            logp, mean, logv, z,prediction = model(batch['input'], batch['length'])
            prediction = prediction.squeeze()

            if opts.data_type == 'physchem':

                temp['score_test'].append(r2_score(batch['physchem'].data.cpu().numpy(),prediction.squeeze().data.cpu().numpy()))

            elif opts.data_type == 'biology':

                temp['AUC_test'].append(roc_auc_score(batch['Effect'].data.cpu().numpy(),prediction.squeeze().data.cpu().numpy()))

            elif opts.data_type == 'physchem_biology':

                temp['score_test'].append(r2_score(batch['physchem'].data.cpu().numpy(),prediction[:,0:-1].squeeze().data.cpu().numpy()))
                temp['AUC_test'].append(roc_auc_score(batch['Effect'].data.cpu().numpy(),prediction[:,-1].squeeze().data.cpu().numpy()))

            
        return temp
    else:
        return None
    
    

def loss_saver(tracker,temp_tr,temp_te,epoch,opts,lr, KL_weight):

    tracker['NLL_loss'].append(np.mean(np.asarray(temp_tr['NLL_loss'])))
    tracker['KL_loss'].append(np.mean(np.asarray(temp_tr['KL_loss'])))
    tracker['ELBO'].append(np.mean(np.asarray(temp_tr['ELBO'])))
        
    if opts.data_type == 'physchem_biology':
        tracker['phys_loss'].append(np.mean(np.asarray(temp_tr['phys_loss'])))
        tracker['bio_loss'].append(np.mean(np.asarray(temp_tr['bio_loss'])))
        tracker['score_train'].append(np.mean(np.asarray(temp_tr['score_train'])))
        tracker['AUC_train'].append(np.mean(np.asarray(temp_tr['AUC_train'])))
        tracker['score_test'].append(np.mean(np.asarray(temp_te['score_test'])))
        tracker['AUC_test'].append(np.mean(np.asarray(temp_te['AUC_test'])))
        print(" hidden %d, Batch %04d/%i, 'ELBO' %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, phys_loss %9.4f,bio_loss %9.4f, AUC_train %9.4f, AUC_test %9.4f, score_train %9.4f,score_test %9.4f, KL-Weight %6.3f, lr %9.7f"
          %(opts.hidden_size, epoch, opts.epochs, 
            np.mean(np.asarray(temp_tr['ELBO'])),
            np.mean(np.asarray(temp_tr['NLL_loss'])),
            np.mean(np.asarray(temp_tr['KL_loss'])),
            np.mean(np.asarray(temp_tr['phys_loss'])),
            np.mean(np.asarray(temp_tr['bio_loss'])),
            np.mean(np.asarray(temp_tr['AUC_train'])),
            np.mean(np.asarray(temp_te['AUC_test'])),
            np.mean(np.asarray(temp_tr['score_train'])),
            np.mean(np.asarray(temp_te['score_test'])),
            KL_weight,
            lr))
    elif opts.data_type == 'physchem':
        tracker['phys_loss'].append(np.mean(np.asarray(temp_tr['phys_loss'])))
        tracker['score_train'].append(np.mean(np.asarray(temp_tr['score_train'])))
        tracker['score_test'].append(np.mean(np.asarray(temp_te['score_test'])))
       
        print(" hidden %d, Batch %04d/%i, 'ELBO' %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, phys_loss %9.4f,score_train %9.4f,score_test %9.4f, KL-Weight %6.3f, lr %9.7f"
          %(opts.hidden_size, epoch, opts.epochs, 
            np.mean(np.asarray(temp_tr['ELBO'])),
            np.mean(np.asarray(temp_tr['NLL_loss'])),
            np.mean(np.asarray(temp_tr['KL_loss'])),
            np.mean(np.asarray(temp_tr['phys_loss'])),
            np.mean(np.asarray(temp_tr['score_train'])),
            np.mean(np.asarray(temp_te['score_test'])),
            KL_weight,
            lr))
        
    elif opts.data_type == 'biology':
        
        tracker['bio_loss'].append(np.mean(np.asarray(temp_tr['bio_loss'])))
        tracker['AUC_train'].append(np.mean(np.asarray(temp_tr['AUC_train'])))
        tracker['AUC_test'].append(np.mean(np.asarray(temp_te['AUC_test'])))
        print(" hidden %d, Batch %04d/%i, 'ELBO' %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f,bio_loss %9.4f, AUC_train %9.4f, AUC_test %9.4f, KL-Weight %6.3f, lr %9.7f"
          %(opts.hidden_size, epoch, opts.epochs, 
            np.mean(np.asarray(temp_tr['ELBO'])),
            np.mean(np.asarray(temp_tr['NLL_loss'])),
            np.mean(np.asarray(temp_tr['KL_loss'])),
            np.mean(np.asarray(temp_tr['bio_loss'])),
            np.mean(np.asarray(temp_tr['AUC_train'])),
            np.mean(np.asarray(temp_te['AUC_test'])),
            KL_weight,lr))
    elif opts.data_type == 'pure_smiles':
        print(" hidden %d, Batch %04d/%i, 'ELBO' %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f,KL-Weight %6.3f, lr %9.7f"
          %(opts.hidden_size, epoch, opts.epochs, 
            np.mean(np.asarray(temp_tr['ELBO'])),
            np.mean(np.asarray(temp_tr['NLL_loss'])),
            np.mean(np.asarray(temp_tr['KL_loss'])),
            KL_weight,lr))
    else:
        raise ValueError('ivalid data type!!!!')
    return tracker 



print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
                  
