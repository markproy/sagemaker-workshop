import os
import argparse
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

def prep_data():
    base_dir = os.environ.get('SM_INPUT_DIR') + '/data'
    import pandas as pd
    xtest_df = pd.read_csv(f'{base_dir}/test/xtest.csv', header=None)
    xtest = xtest_df.values
    xtrain_df = pd.read_csv(f'{base_dir}/train/xtrain.csv', header=None)
    xtrain = xtrain_df.values
    xval_df = pd.read_csv(f'{base_dir}/val/xval.csv', header=None)
    xval = xval_df.values

    ytest_df = pd.read_csv(f'{base_dir}/test/ytest.csv', header=None)
    ytest = ytest_df.values
    ytrain_df = pd.read_csv(f'{base_dir}/train/ytrain.csv', header=None)
    ytrain = ytrain_df.values
    yval_df = pd.read_csv(f'{base_dir}/val/yval.csv', header=None)
    yval = yval_df.values

    print('xtr: {}, xte: {}, xv: {}, ytr: {}, yte: {}, yv: {}'.format(xtrain.shape, 
                                                                      xtest.shape, xval.shape,
                                                                      ytrain.shape, ytest.shape,
                                                                      yval.shape))

    return xtrain, xtest, xval, ytrain, ytest, yval
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dense1_dim', type=int, default=16)
    parser.add_argument('--dense1_activation', type=str, default='relu')
    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))

    args, _ = parser.parse_known_args()
    print('args: {}'.format(args))
    epochs = args.epochs
    dense1_dim = args.dense1_dim
    dense1_activation = args.dense1_activation
    
    X_train, X_test, X_val, y_train, y_test, y_val = prep_data()
    
    network = models.Sequential()
    network.add(layers.Dense(dense1_dim, activation=dense1_activation, 
                             input_shape=(13,)))
    network.add(layers.Dense(16, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))
    network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    network.fit(X_train, y_train, epochs=epochs, batch_size=100,
                         validation_data=(X_val, y_val))
    print('\nTraining completed.')
    print('\nEvaluating against test set...')
    network.evaluate(x=X_test, y=y_test)
    print('\nSaving model')
    
    model_version = '1'
    export_dir = os.environ.get('SM_MODEL_DIR') + '/' + 'export/Servo/' + model_version
    print('Export dir: ' + export_dir)

#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
#https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
    
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_dir,
            inputs={'input_data': network.input},
            outputs={'score': network.output})

# Detailed calls fail...
#tensorflow.python.framework.errors_impl.FailedPreconditionError: Error while reading resource variable dense_2_1/bias from Container: localhost. This could mean that the variable was uninitialized. Not found: Container localhost does not exist. (Could not find resource: localhost/dense_2_1/bias)

    print('\nExiting training script.')
