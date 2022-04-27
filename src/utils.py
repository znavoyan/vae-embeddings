import numpy as np
import pandas as pd
import json
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
import os
import ast
from pathlib import Path
from rdkit import Chem
import tensorflow as tf


def accuracy_logS(y_true, y_pred):
    correct = 0
    for t, p in zip(y_true, y_pred):
        if (p >= t - 0.7) and (p <= t + 0.7):
            correct += 1
    return correct/len(y_true)*100


def percent_diff(x1, x2):
    return abs(x1 - x2)/abs(np.mean([x1, x2]))*100


def canon_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
    return smiles


def split_into_folds(data_path, num_folds, num_repeats, property, save_path):
    data_df = pd.read_csv(data_path)
    print('Dataset size: ', data_df.shape[0])
    #compound_indices = data_df['InchiKey'].values
    compound_indices = list(data_df.index)

    print(f'Splitting the data into {num_folds} folds...')
    if property == 'logS':
        kf = RepeatedKFold(n_splits = num_folds, n_repeats = num_repeats, random_state = 1)
        y = data_df.new_logS.values
        kf.get_n_splits(compound_indices)

    elif property == 'logD':
        kf = RepeatedKFold(n_splits = num_folds, n_repeats = num_repeats, random_state = 1)
        y = data_df.new_logD.values
        kf.get_n_splits(compound_indices)

    elif property == 'logBB':
        kf = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = num_repeats, random_state = 1)
        y = data_df.new_BBclass.values
        kf.get_n_splits(compound_indices, y)

    for i in range(num_folds*num_repeats):
        Path(f"{save_path}/fold{i+1}").mkdir(parents=True, exist_ok=True)

    fold_idx = 1

    for train_index, test_index in kf.split(compound_indices, y):
        np.save(os.path.join(save_path, f'fold{fold_idx}/train_index'), train_index)
        np.save(os.path.join(save_path, f'fold{fold_idx}/test_index'), test_index)
        print('Fold: ', fold_idx)
        print('Training size: ', len(train_index), '\tTesting size: ', len(test_index))
        fold_idx += 1


def load_fold_data(data_path, fold_num, feature, fold_indices_path, property):
    data_df = pd.read_csv(data_path)
    train_index = np.load(f'{fold_indices_path}/fold{fold_num}/train_index.npy')
    test_index = np.load(f'{fold_indices_path}/fold{fold_num}/test_index.npy')

    train_df = data_df.iloc[train_index]
    test_df = data_df.iloc[test_index]

    train_data = train_df[feature].values
    test_data = test_df[feature].values
    X_train = np.array([ast.literal_eval(i) for i in train_data], dtype=float)
    X_test = np.array([ast.literal_eval(i) for i in test_data], dtype=float)

    if property == 'logS':
        y_train = np.array(train_df.new_logS.values)
        y_test = np.array(test_df.new_logS.values)
    if property == 'logD':
        y_train = np.array(train_df.new_logD.values)
        y_test = np.array(test_df.new_logD.values)
    elif property == 'logBB':
        y_train = np.array(train_df.new_BBclass.values)
        y_test = np.array(test_df.new_BBclass.values)

    return X_train, X_test, y_train, y_test

'''
def load_full_data(data_path, feature):
    train_df = pd.read_csv(data_path)

    train_data = train_df[feature].values
    X_train = np.array([ast.literal_eval(i) for i in train_data], dtype=float)
    y_train = np.array(train_df.logS.values)

    return X_train, y_train'''


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_tf(features, labels, alpha=0.2):
    # tensorflow version
    print("features",features.shape)
    num_examples = features.shape[0]
    mix = tf.distributions.Beta(alpha, alpha).sample([num_examples, 1, 1])
    #mix = tf.maximum(mix, 1 - mix)
    features = features * mix + features[::-1] * (1 - mix)
    labels = labels * mix[:, 0] + labels[::-1] * (1 - mix[:, 0])
    return features, labels

def mixup_np(features, labels, alpha=0.1):
    # numpy version
    num_examples = features.shape[0]
    num_class = labels.shape[-1]
    mix = np.random.beta(alpha, alpha, size=[num_examples])
    features = np.swapaxes(features, 0, -1)
    features = features * mix + features[::-1] * (np.ones_like(mix) - mix)
    features = np.swapaxes(features, 0, -1)

    labels = np.swapaxes(labels, 0, -1)
    labels = labels * mix + labels[::-1] * (np.ones_like(mix) - mix)
    labels = np.swapaxes(labels, 0, -1)
    
    return features, labels