import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from resnet import ResNetLogS
import ast
import json
import os
import sys
sys.path.append('/home/ani/projects/final_project/src')
from utils import load_fold_data
from pathlib import Path
from sklearn.metrics import r2_score, f1_score, roc_auc_score, accuracy_score
from rdkit import Chem
import rdkit.Chem.Descriptors as descriptors

def calculate_logP(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return descriptors.MolLogP(mol)
    except Exception as e:
        print(e)
        return np.nan

#parser = argparse.ArgumentParser()
#parser.add_argument('--experiment', default = '../experiments/(sc6)_2021-03-22_22:45:17_cv_fing_emb_withoutdots_loss_rmse_lr_1e-05')
#args = parser.parse_args()

#print(args)
experiment = '/home/ani/projects/final_project/models_fixed/pvae/cv10_ResNet20_logBB_2906_vae_emb_tpsa_196_3'
# data_path = '/home/ani/projects/final_project/data/logS/processed_with_vae/novel_62_compounds_vae_logp_62.csv'

# Path(f"{experiment}/predictions").mkdir(parents=True, exist_ok=True)

with open(os.path.join(experiment, 'config.json')) as f:
    params = json.load(f)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

data_path = params['data']
print(data_path)
print('Feature: ', params['feature'])

data_df = pd.read_csv(data_path)
#data_df[params['feature']] = data_df[params['feature']].apply(ast.literal_eval)
data_df.vae_emb = data_df.vae_emb.apply(ast.literal_eval)
# data_df.fingerprint = data_df.fingerprint.apply(ast.literal_eval)
# data_df.fing_emb = data_df.fing_emb.apply(ast.literal_eval)

# data_df['logP'] = data_df.SMILES.apply(calculate_logP)
# data_df['vae_emb_desc'] = [i + [j] for i, j in zip(data_df.vae_emb.values, data_df.logP.values)]
# data_df['fing_emb_desc'] = [i + [j] for i, j in zip(data_df.fing_emb.values, data_df.logP.values)]
# data_df['fing_desc'] = [i + [j] for i, j in zip(data_df.fingerprint.values, data_df.logP.values)]

# print(data_df[data_df.isna()])



print(data_df.shape)
input_len = len(data_df[params['feature']].iloc[0])
  
# r2_list = []
acc_list = []
f1_list = []
auc_list = []

print('Predicting...')
for fold in range(1, 11):
    test_index = np.load(os.path.join(params['fold_indices_dir'], f'fold{fold}/test_index.npy'))

    X_test = np.array(list(data_df.iloc[test_index][params['feature']].values), dtype=float)
    y_test = np.array(data_df.iloc[test_index].new_BBclass.values)

    tf.reset_default_graph()
    model = ResNetLogS(input_len)

    X=tf.placeholder(tf.float32, [None, input_len])
    y=tf.placeholder(tf.float32, [None, 1])

    preds = tf.round(tf.nn.sigmoid(model.forward(X)))
    
    correct_pred = tf.equal(preds , y)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print('Fold: ', fold)
    #preds_df = pd.DataFrame({'SMILES': test_smiles, 'true_logS': y_test})

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(experiment, f'models/fold{fold}/model-85'))

        predictions = sess.run(preds, feed_dict={X:X_test})
        #preds_df['prdicted_logS'] = predictions

        # r2 = r2_score(y_true = y_test, y_pred = predictions)
        # r2_list.append(r2)
        print(sess.run(acc, feed_dict={X:X_test, y: y_test.reshape([-1, 1])}))
        # acc_list.append(sess.run(acc, feed_dict={X:X_test, y: y_test.reshape([-1, 1])}))
        acc_list.append(accuracy_score(y_true = y_test, y_pred = predictions))
        f1_list.append(f1_score(y_true = y_test, y_pred = predictions))
        auc_list.append(roc_auc_score(y_true = y_test, y_score = predictions))

        #preds_df.to_csv(f'{experiment}/predictions/prediction_fold{fold}.csv', index = False)
# print('Mean R^2: ', np.mean(r2_list))
print('Mean acc: ', np.mean(acc_list), np.std(acc_list))
print('Mean f1: ', np.mean(f1_list), np.std(f1_list))
print('Mean auc: ', np.mean(auc_list), np.std(auc_list))