import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.metrics import accuracy
import argparse
from resnet import ResNet20
import ast

import sys
from utils import load_fold_data

from tqdm import tqdm
import os
import json
from pathlib import Path
from datetime import datetime
from utils import accuracy_logS, split_into_folds
import logging
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import joblib



def train_logS_resnet(params):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    LOG = os.path.join(params.save_dir, f"train_log_{params.feature}.log")
    logging.basicConfig(filename=LOG, filemode='w', level=logging.DEBUG)


    with open(f"{params.save_dir}/config.json", "w") as outfile:
        json.dump(vars(params), outfile, indent=4)

    num_folds = len(os.listdir(params.fold_indices_dir))
    data_df = pd.read_csv(params.data)
    input_len = len(ast.literal_eval(data_df[params.feature].iloc[0]))
    print('Input length:', input_len)
    logging.info('Input length: ' + str(input_len))

    fold_train_r2 = []
    fold_test_r2 = []
    fold_train_rmse = []
    fold_test_rmse = []
    fold_train_acc = []
    fold_test_acc = []


    for fold_num in range(params.start_fold, num_folds + 1):

        train_loss_hist = []
        test_loss_hist = []
        train_r2_hist = []
        test_r2_hist = []
        train_rmse_hist = []
        test_rmse_hist = []
        train_acc_hist = []
        test_acc_hist = []

        tf.reset_default_graph()

        X=tf.placeholder(tf.float32, [None, input_len])
        y=tf.placeholder(tf.float32, [None, 1])

        # Build model
        model = ResNet20(input_len)
        preds = model.forward(X)
        logging.info(f'Model trainable parameters count: {model.count_model_params()}')

        # L2 weight decay
        model_weights = [v for v in tf.trainable_variables() if 'w_fc' in v.name or 'kernel' in v.name]
        weights_l2 = [params.l2_wd*tf.nn.l2_loss(i) for i in model_weights]

        # Root Mean Squared Error
        rmse = tf.sqrt(tf.reduce_mean((y - preds)**2))
        # R-squared
        r2_numinator = tf.reduce_sum((y - preds)**2)
        r2_denominator = tf.reduce_sum((y - tf.reduce_mean(y))**2)
        r2_score = 1 - r2_numinator/r2_denominator
        # Loss function` RMSE + L2-weight-decay
        loss = rmse + tf.reduce_sum(weights_l2)

        # Collect summaries for Tensorboard
        tf_rmse_summary = tf.summary.scalar('loss', loss)
        tf_r2_summary = tf.summary.scalar('r2', r2_score)
        performance_summaries = tf.summary.merge([tf_r2_summary,tf_rmse_summary])

        optimizer=tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            summaries_dir = f'{experiment_name}/logs/fold{fold_num}'
            train_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test'), sess.graph)

            init=tf.global_variables_initializer()
            sess.run(init)

            print(f'Loading the data of fold {fold_num}...')
            logging.info(f'Loading the data of fold {fold_num}...')
            X_train, X_test, y_train, y_test = load_fold_data(params.data, fold_num, params.feature, params.fold_indices_dir, params.property)

            print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
            logging.info(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')

            for epoch in range(1, params.epochs + 1):
                print(f'Epoch: {epoch}/{params.epochs}')
                logging.info(f'Epoch: {epoch}/{params.epochs}')

                for num_batch in tqdm(range(len(X_train)//params.batch_size+1)):
                    train_data = X_train[num_batch*params.batch_size : min((num_batch+1)*params.batch_size, len(X_train))]
                    train_labels = y_train[num_batch*params.batch_size : min((num_batch+1)*params.batch_size, len(y_train))]

                    batch_preds, _ = sess.run([preds, optimizer],  feed_dict={X:train_data, y:train_labels.reshape([-1, 1])})

                summary, train_predictions, train_loss, \
                epoch_train_rmse, epoch_train_r2 = sess.run([performance_summaries, preds, loss, rmse, r2_score],
                                                            feed_dict = {X: X_train, y: y_train.reshape([-1, 1])})
                train_writer.add_summary(summary, epoch)
                train_accuracy = accuracy_logS(y_true=y_train, y_pred=train_predictions)
                print(f'train_r2: {epoch_train_r2}, train_rmse: {epoch_train_rmse}, train_acc: {train_accuracy}')
                logging.info(f'train_r2: {epoch_train_r2}, train_rmse: {epoch_train_rmse}, train_acc: {train_accuracy}')

                train_loss_hist.append(train_loss)
                train_r2_hist.append(epoch_train_r2)
                train_rmse_hist.append(epoch_train_rmse)
                train_acc_hist.append(train_accuracy)

                summary, test_predictions, test_loss, test_rmse, test_r2 = sess.run([performance_summaries, preds, loss, rmse, r2_score],
                                              feed_dict={X:X_test, y:y_test.reshape([-1, 1])})
                test_accuracy = accuracy_logS(y_true=y_test, y_pred=test_predictions)
                test_writer.add_summary(summary, epoch)
                print(f'test_r2: {test_r2}, test_rmse: {test_rmse}, test_acc: {test_accuracy}')
                logging.info(f'test_r2: {test_r2}, test_rmse: {test_rmse}, test_acc: {test_accuracy}')

                test_loss_hist.append(test_loss)
                test_r2_hist.append(test_r2)
                test_rmse_hist.append(test_rmse)
                test_acc_hist.append(test_accuracy)

                if epoch % params.epochs == 0:
                    model_path = f'{experiment_name}/models/fold{fold_num}/model'
                    print(f'Saving the model to {model_path}')
                    logging.info(f'Saving the model to {model_path}')
                    saver.save(sess, model_path, global_step=epoch)


            history_df = pd.DataFrame({'epoch': np.arange(1, params.epochs + 1), 
                                       'train_loss': train_loss_hist, 'test_loss': test_loss_hist,
                                       'train_r2': train_r2_hist, 'test_r2': test_r2_hist,
                                       'train_rmse': train_rmse_hist, 'test_rmse': test_rmse_hist,
                                       'train_acc': train_acc_hist, 'test_acc': test_acc_hist})

            history_df.to_csv(f'{experiment_name}/histories/history_fold{fold_num}.csv', index = False)

            fold_train_r2.append(train_r2_hist[-1])
            fold_test_r2.append(test_r2_hist[-1])
            fold_train_rmse.append(train_rmse_hist[-1])
            fold_test_rmse.append(test_rmse_hist[-1])
            fold_train_acc.append(train_acc_hist[-1])
            fold_test_acc.append(test_acc_hist[-1])

    pd.DataFrame({'fold': np.arange(1, num_folds + 1), 
                  'train_r2': fold_train_r2, 'test_r2': fold_test_r2,
                  'train_rmse': fold_train_rmse, 'test_rmse': fold_test_rmse,
                  'train_acc': fold_train_acc, 'test_acc': fold_test_acc}).to_csv(f'{experiment_name}/final_metrics_each_fold.csv')



def train_logBB_resnet(params):    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    LOG = os.path.join(params.save_dir, f"train_log_{params.feature}.log")
    logging.basicConfig(filename=LOG, filemode='w', level=logging.DEBUG)

    with open(f"{params.save_dir}/config.json", "w") as outfile:
        json.dump(vars(params), outfile, indent=4)

    num_folds = len(os.listdir(params.fold_indices_dir))
    data_df = pd.read_csv(params.data)
    input_len = len(ast.literal_eval(data_df[params.feature].iloc[0]))
    print('Input length:', input_len)
    logging.info('Input length: ' + str(input_len))


    fold_train_loss = []
    fold_test_loss = []
    fold_train_acc = []
    fold_test_acc = []

    for fold_num in range(params.start_fold, num_folds + 1):

        train_loss_hist = []
        test_loss_hist = []
        train_acc_hist = []
        test_acc_hist = []

        tf.reset_default_graph()

        X=tf.placeholder(tf.float32, [None, input_len])
        y=tf.placeholder(tf.float32, [None, 1])

        # Build model
        model = ResNet20(input_len)
        preds = model.forward(X)
        logging.info(f'Model trainable parameters count: {model.count_model_params()}')

        # L2 weight decay
        model_weights = [v for v in tf.trainable_variables() if 'w_fc' in v.name or 'kernel' in v.name]
        weights_l2 = [params.l2_wd*tf.nn.l2_loss(i) for i in model_weights]

        # Binary Cross Entropy (with sigmoid)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=preds))
        loss = cross_entropy + tf.reduce_sum(weights_l2)
        
        predicted = tf.nn.sigmoid(preds)
        correct_pred = tf.equal(tf.round(predicted), y)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Collect summaries for Tensorboard
        tf_loss_summary = tf.summary.scalar('loss', loss)
        tf_acc_summary = tf.summary.scalar('acc', acc)
        performance_summaries = tf.summary.merge([tf_loss_summary, tf_acc_summary])

        optimizer=tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            summaries_dir = f'{experiment_name}/logs/fold{fold_num}'
            train_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test'), sess.graph)

            init = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            sess.run(init)
            sess.run(init_l)

            print(f'Loading the data of fold {fold_num}...')
            logging.info(f'Loading the data of fold {fold_num}...')
            X_train, X_test, y_train, y_test = load_fold_data(params.data, fold_num, params.feature, params.fold_indices_dir, params.property)

            print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
            logging.info(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')

            for epoch in range(1, params.epochs + 1):
                print(f'Epoch: {epoch}/{params.epochs}')
                logging.info(f'Epoch: {epoch}/{params.epochs}')

                for num_batch in tqdm(range(len(X_train)//params.batch_size+1)):
                    train_data = X_train[num_batch*params.batch_size : min((num_batch+1)*params.batch_size, len(X_train))]
                    train_labels = y_train[num_batch*params.batch_size : min((num_batch+1)*params.batch_size, len(y_train))]

                    batch_preds, _ = sess.run([preds, optimizer],
                                                          feed_dict={X:train_data, y:train_labels.reshape([-1, 1])})

                summary, train_predictions, train_loss, train_acc = sess.run([performance_summaries, preds, loss, acc],
                                                            feed_dict = {X: X_train, y: y_train.reshape([-1, 1])})

                train_writer.add_summary(summary, epoch)
                print(f'train_loss: {train_loss}, train_acc: {train_acc}')
                logging.info(f'train_loss: {train_loss}, train_acc: {train_acc}')

                train_loss_hist.append(train_loss)
                train_acc_hist.append(train_acc)

                summary, test_predictions, test_loss, test_acc = sess.run([performance_summaries, preds, loss, acc],
                                              feed_dict={X:X_test, y:y_test.reshape([-1, 1])})
                
                test_writer.add_summary(summary, epoch)
                print(f'test_loss: {test_loss}, test_acc: {test_acc}')
                logging.info(f'test_loss: {test_loss}, test_acc: {test_acc}')

                test_loss_hist.append(test_loss)
                test_acc_hist.append(test_acc)

                # Saving the model after the last epoch
                if epoch % params.epochs == 0:
                    model_path = f'{experiment_name}/models/fold{fold_num}/model'
                    print(f'Saving the model to {model_path}')
                    logging.info(f'Saving the model to {model_path}')
                    saver.save(sess, model_path, global_step=epoch)

            history_df = pd.DataFrame({'train_loss': train_loss_hist, 'test_loss': test_loss_hist,
                                       'train_acc': train_acc_hist, 'test_acc': test_acc_hist})

            history_df.to_csv(f'{experiment_name}/histories/history_fold{fold_num}.csv', index = False)

            fold_train_loss.append(train_loss_hist[-1])
            fold_test_loss.append(test_loss_hist[-1])
            fold_train_acc.append(train_acc_hist[-1])
            fold_test_acc.append(test_acc_hist[-1])

    pd.DataFrame({'fold': np.arange(1, num_folds + 1),
                  'train_rmse': fold_train_loss, 'test_rmse': fold_test_loss,
                  'train_acc': fold_train_acc, 'test_acc': fold_test_acc}).to_csv(f'{experiment_name}/final_metrics_each_fold.csv')


def train_logD_resnet(params):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    LOG = os.path.join(params.save_dir, f"train_log_{params.feature}.log")
    logging.basicConfig(filename=LOG, filemode='w', level=logging.DEBUG)


    with open(f"{params.save_dir}/config.json", "w") as outfile:
        json.dump(vars(params), outfile, indent=4)

    num_folds = len(os.listdir(params.fold_indices_dir))
    data_df = pd.read_csv(params.data)
    input_len = len(ast.literal_eval(data_df[params.feature].iloc[0]))
    print('Input length:', input_len)
    logging.info('Input length: ' + str(input_len))

    fold_train_r2 = []
    fold_test_r2 = []
    fold_train_rmse = []
    fold_test_rmse = []


    for fold_num in range(params.start_fold, num_folds + 1):

        train_loss_hist = []
        test_loss_hist = []
        train_r2_hist = []
        test_r2_hist = []
        train_rmse_hist = []
        test_rmse_hist = []

        tf.reset_default_graph()

        X=tf.placeholder(tf.float32, [None, input_len])
        y=tf.placeholder(tf.float32, [None, 1])

        # Build model
        model = ResNet20(input_len)
        preds = model.forward(X)
        logging.info(f'Model trainable parameters count: {model.count_model_params()}')

        # L2 weight decay
        model_weights = [v for v in tf.trainable_variables() if 'w_fc' in v.name or 'kernel' in v.name]
        weights_l2 = [params.l2_wd*tf.nn.l2_loss(i) for i in model_weights]

        # Root Mean Squared Error
        rmse = tf.sqrt(tf.reduce_mean((y - preds)**2))
        # R-squared
        r2_numinator = tf.reduce_sum((y - preds)**2)
        r2_denominator = tf.reduce_sum((y - tf.reduce_mean(y))**2)
        r2_score = 1 - r2_numinator/r2_denominator
        # Loss function` RMSE + L2-weight-decay
        loss = rmse + tf.reduce_sum(weights_l2)

        # Collect summaries for Tensorboard
        tf_rmse_summary = tf.summary.scalar('loss', loss)
        tf_r2_summary = tf.summary.scalar('r2', r2_score)
        performance_summaries = tf.summary.merge([tf_r2_summary,tf_rmse_summary])

        optimizer=tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            summaries_dir = f'{experiment_name}/logs/fold{fold_num}'
            train_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test'), sess.graph)

            init=tf.global_variables_initializer()
            sess.run(init)

            print(f'Loading the data of fold {fold_num}...')
            logging.info(f'Loading the data of fold {fold_num}...')
            X_train, X_test, y_train, y_test = load_fold_data(params.data, fold_num, params.feature, params.fold_indices_dir, params.property)

            print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
            logging.info(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')

            for epoch in range(1, params.epochs + 1):
                print(f'Epoch: {epoch}/{params.epochs}')
                logging.info(f'Epoch: {epoch}/{params.epochs}')

                for num_batch in tqdm(range(len(X_train)//params.batch_size+1)):
                    train_data = X_train[num_batch*params.batch_size : min((num_batch+1)*params.batch_size, len(X_train))]
                    train_labels = y_train[num_batch*params.batch_size : min((num_batch+1)*params.batch_size, len(y_train))]

                    batch_preds, _ = sess.run([preds, optimizer],
                                                          feed_dict={X:train_data, y:train_labels.reshape([-1, 1])})

                summary, train_predictions, train_loss, \
                epoch_train_rmse, epoch_train_r2 = sess.run([performance_summaries, preds, loss, rmse, r2_score],
                                                            feed_dict = {X: X_train, y: y_train.reshape([-1, 1])})
                train_writer.add_summary(summary, epoch)
                print(f'train_r2: {epoch_train_r2}, train_rmse: {epoch_train_rmse}')
                logging.info(f'train_r2: {epoch_train_r2}, train_rmse: {epoch_train_rmse}')

                train_loss_hist.append(train_loss)
                train_r2_hist.append(epoch_train_r2)
                train_rmse_hist.append(epoch_train_rmse)

                summary, test_predictions, test_loss, test_rmse, test_r2 = sess.run([performance_summaries, preds, loss, rmse, r2_score],
                                              feed_dict={X:X_test, y:y_test.reshape([-1, 1])})
                test_writer.add_summary(summary, epoch)
                print(f'test_r2: {test_r2}, test_rmse: {test_rmse}')
                logging.info(f'test_r2: {test_r2}, test_rmse: {test_rmse}')

                test_loss_hist.append(test_loss)
                test_r2_hist.append(test_r2)
                test_rmse_hist.append(test_rmse)

                if epoch % params.epochs == 0:
                    model_path = f'{experiment_name}/models/fold{fold_num}/model'
                    print(f'Saving the model to {model_path}')
                    logging.info(f'Saving the model to {model_path}')
                    saver.save(sess, model_path, global_step=epoch)


            history_df = pd.DataFrame({'epoch': np.arange(1, params.epochs + 1), 
                                       'train_loss': train_loss_hist, 'test_loss': test_loss_hist,
                                       'train_r2': train_r2_hist, 'test_r2': test_r2_hist,
                                       'train_rmse': train_rmse_hist, 'test_rmse': test_rmse_hist})

            history_df.to_csv(f'{experiment_name}/histories/history_fold{fold_num}.csv', index = False)

            fold_train_r2.append(train_r2_hist[-1])
            fold_test_r2.append(test_r2_hist[-1])
            fold_train_rmse.append(train_rmse_hist[-1])
            fold_test_rmse.append(test_rmse_hist[-1])

    pd.DataFrame({'fold': np.arange(1, num_folds + 1), 
                  'train_r2': fold_train_r2, 'test_r2': fold_test_r2,
                  'train_rmse': fold_train_rmse, 'test_rmse': fold_test_rmse}).to_csv(f'{experiment_name}/final_metrics_each_fold.csv')


def train_mlp(params):
    num_folds = len(os.listdir(params.fold_indices_dir))
    Path(f"{params.save_dir}/models/mlp").mkdir(parents=True, exist_ok=True)

    if params.property.lower() == 'logs' or params.property.lower() == 'logd':
        r2_list = []
        rmse_list = []

        for fold_num in range(params.start_fold, num_folds + 1):
            X_train, X_test, y_train, y_test = load_fold_data(params.data, fold_num, params.feature, params.fold_indices_dir, params.property)

            model = MLPRegressor(solver = 'sgd', max_iter = params.mlp_max_iter).fit(X_train, y_train)
            filename = os.path.join(params.save_dir, f'/models/mlp/mlp_fold{i}.sav')
            joblib.dump(model, filename)

            preds = model.predict(X_test)
            r2_list.append(r2_score(y_test, preds))
            rmse_list.append(mean_squared_error(y_test, preds, squared = False))

        pd.DataFrame({'fold': np.arange(1, num_folds + 1), 
                  'test_r2': r2_list,
                  'test_rmse': rmse_list}).to_csv(f'{params.save_dir}/final_metrics_mlp_each_fold.csv')

    elif params.property.lower() == 'logbb':
        acc_list = []
        f1_list = []
        auc_list = []
        for fold_num in range(params.start_fold, num_folds + 1):
            X_train, X_test, y_train, y_test = load_fold_data(params.data, fold_num, params.feature, params.fold_indices_dir, params.property)

            model = MLPClassifier(solver = 'sgd', max_iter = params.mlp_max_iter).fit(X_train, y_train)
            filename = os.path.join(params.save_dir, f'/models/mlp/mlp_fold{i}.sav')
            joblib.dump(model, filename)

            preds = model.predict(X_test)
            acc_list.append(accuracy_score(y_true = y_test, y_pred = preds))
            f1_list.append(f1_score(y_true = y_test, y_pred = preds))
            auc_list.append(roc_auc_score(y_true = y_test, y_score = preds))

        pd.DataFrame({'fold': np.arange(1, num_folds + 1), 
                  'test_acc': acc_list,
                  'test_f1': f1_list,
                  'test_auc': auc_list}).to_csv(f'{params.save_dir}/final_metrics_mlp_each_fold.csv')


def train_lr(params):
    num_folds = len(os.listdir(params.fold_indices_dir))
    Path(f"{params.save_dir}/models/lr").mkdir(parents=True, exist_ok=True)

    if params.property.lower() == 'logs' or params.property.lower() == 'logd':
        r2_list = []
        rmse_list = []

        for fold_num in range(params.start_fold, num_folds + 1):
            X_train, X_test, y_train, y_test = load_fold_data(params.data, fold_num, params.feature, params.fold_indices_dir, params.property)

            model = LinearRegression(solver = 'sgd').fit(X_train, y_train)
            filename = os.path.join(params.save_dir, f'/models/lr/lr_fold{i}.sav')
            joblib.dump(model, filename)

            preds = model.predict(X_test)
            r2_list.append(r2_score(y_test, preds))
            rmse_list.append(mean_squared_error(y_test, preds, squared = False))

        pd.DataFrame({'fold': np.arange(1, num_folds + 1), 
                  'test_r2': r2_list,
                  'test_rmse': rmse_list}).to_csv(f'{params.save_dir}/final_metrics_lr_each_fold.csv')

    elif params.property.lower() == 'logbb':
        acc_list = []
        f1_list = []
        auc_list = []
        for fold_num in range(params.start_fold, num_folds + 1):
            X_train, X_test, y_train, y_test = load_fold_data(params.data, fold_num, params.feature, params.fold_indices_dir, params.property)

            model = LogisticRegression(max_iter = 200).fit(X_train, y_train)
            filename = os.path.join(params.save_dir, f'/models/lr/lr_fold{i}.sav')
            joblib.dump(model, filename)

            preds = model.predict(X_test)
            acc_list.append(accuracy_score(y_true = y_test, y_pred = preds))
            f1_list.append(f1_score(y_true = y_test, y_pred = preds))
            auc_list.append(roc_auc_score(y_true = y_test, y_score = preds))

        pd.DataFrame({'fold': np.arange(1, num_folds + 1), 
                  'test_acc': acc_list,
                  'test_f1': f1_list,
                  'test_auc': auc_list}).to_csv(f'{params.save_dir}/final_metrics_lr_each_fold.csv')


def train(params):
    if params.model.lower() == 'resnet':
        if params.property.lower() == 'logs':
            train_logS_resnet(params)
        if params.property.lower() == 'logd':
            train_logD_resnet(params)
        if params.property.lower() == 'logbb':
            train_logBB_resnet(params)

    elif params.model.lower() == 'mlp':
        train_mlp(params)

    elif params.model.lower() == 'lr':
        train_lr(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--property', choices = ['logS', 'logBB', 'logD'])
    parser.add_argument('--data')
    parser.add_argument('--save_dir')
    parser.add_argument('--feature', choices=['fingerprint', 'vae_emb', 'fing_emb', 'fing_desc', 'vae_emb_desc', 'fing_emb_desc'])
    parser.add_argument('--fold_num', type  = int, default = 10)
    parser.add_argument('--repeat_folds', type = int, default = 1)
    parser.add_argument('--fold_indices_dir', help = 'Path to folder containing train-test indices for each fold')
    parser.add_argument('--start_fold', type = int, default = 1)
    parser.add_argument('--model', choices = ['ResNet', 'MLP', 'LR'])

    parser.add_argument('--mlp_max_iter', type=int, default = None)
    parser.add_argument('--epochs', type=int, default = None)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=47)
    parser.add_argument('--l2_wd', type = float, default=0.00001, help='L2 weight decay')
    args = parser.parse_args()

    experiment_name = args.save_dir
    Path(experiment_name).mkdir(parents=True, exist_ok=True)
    Path(f"{experiment_name}/logs").mkdir(parents=True, exist_ok=True)
    Path(f"{experiment_name}/models").mkdir(parents=True, exist_ok=True)
    Path(f"{experiment_name}/histories").mkdir(parents=True, exist_ok=True)

    for i in range(10):
        Path(f"{experiment_name}/models/fold{i+1}").mkdir(parents=True, exist_ok=True)
        Path(f"{experiment_name}/logs/fold{i+1}").mkdir(parents=True, exist_ok=True)

    if (not os.path.isdir(args.fold_indices_dir)) or (len(os.listdir(args.fold_indices_dir)) == 0):
        split_into_folds(args.data, args.fold_num, args.repeat_folds, args.property, args.fold_indices_dir)

    if args.property.lower() == 'logs':
        args.loss = 'rmse'
        if not args.epochs:
            args.epochs = 2000
        if not args.mlp_max_iter:
            args.mlp_max_iter = 700
        train(args)

    elif args.property.lower() == 'logd':
        args.loss = 'rmse'
        if not args.epochs:
            args.epochs = 1500
        if not args.mlp_max_iter:
            args.mlp_max_iter = 1000
        train(args)

    elif args.property.lower() == 'logbb':
        args.loss = 'binary_cross_entropy'
        if not args.epochs:
            args.epochs = 85
        if not args.mlp_max_iter:
            args.mlp_max_iter = 1000    
        train(args)

    else:
        raise ValueError('Invalid property name!')