#     Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License").
#     You may not use this file except in compliance with the License.
#     A copy of the License is located at
#
#         https://aws.amazon.com/apache-2-0/
#
#     or in the "license" file accompanying this file. This file is distributed
#     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#     express or implied. See the License for the specific language governing
#     permissions and limitations under the License.

import os
import argparse
from os import listdir
from os.path import isfile, join

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np
import logging
import json
import glob
import datetime


logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

NUM_FEATURES = 7499 #7499 
NUM_DATA_BATCHES = 5
INPUT_TENSOR_NAME = 'inputs_input'  # needs to match the name of the first layer + "_input"

def list_files_in_dir(which_dir):
    logging.info('\nState of directory tree {}:'.format(which_dir))
    for filename in glob.iglob(which_dir + '**/*', recursive=True):
        logging.info(filename)

class PipeDebugCallback(tf.keras.callbacks.Callback):
  def on_train_batch_begin(self, batch, logs=None):
    logging.info('Training: batch {} BEGINS at {}'.format(batch, datetime.datetime.now().time()))
    list_files_in_dir('/opt/ml/input/data')

  def on_train_batch_end(self, batch, logs=None):
    logging.info('Training: batch {} ENDS at {}'.format(batch, datetime.datetime.now().time()))
    list_files_in_dir('/opt/ml/input/data')

  def on_test_batch_begin(self, batch, logs=None):
    logging.info('Evaluating: batch {} BEGINS at {}'.format(batch, datetime.datetime.now().time()))
    list_files_in_dir('/opt/ml/input/data')

  def on_test_batch_end(self, batch, logs=None):
    logging.info('Evaluating: batch {} ENDS at {}'.format(batch, datetime.datetime.now().time()))
    list_files_in_dir('/opt/ml/input/data')
    
def get_filenames(channel_name, channel):
    list_files_in_dir('/opt/ml/input/data')
    mode = args.data_config[channel_name]['TrainingInputMode']
    fnames = []
    if channel_name in ['train', 'val', 'test']:
        if mode == 'File':
            for f in listdir(channel):
                fnames.append(os.path.join(channel, f))
#        else:
            # in Pipe mode, the files are supposedly in channel_epochnum
            # all seems to work, however, found nothing in when called at start:
            #     /opt/ml/input/data/train/train, /opt/ml/input/data/train/train_0,
            #     /opt/ml/input/data/train_0
        
            #channel_epoch = os.path.join(channel, channel_name + '_0')
            #for f in listdir(channel_epoch):
            #    fnames.append(os.path.join(channel_epoch, f))
            
        logging.info('returning filenames: {}'.format(fnames))
        return [fnames]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)

def train_input_fn():
    return _input(args.epochs, args.batch_size, args.train, 'train')

def test_input_fn():
    return _input(args.epochs, args.batch_size, args.test, 'test')

def val_input_fn():
    return _input(args.epochs, args.batch_size, args.val, 'val')


def _dataset_parser(value):
    """Parse a record from 'value'."""
    feature_description = {
        'features': tf.VarLenFeature(tf.float32),
        'label'   : tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.parse_single_example(value, feature_description)

    label = tf.cast(example['label'], tf.int32)
    logging.info('parsed label: {}'.format(label))
    data  = example['features'].values 
    logging.info('parsed features: {}'.format(data))
    
    return data, label


def _input(epochs, batch_size, channel, channel_name):
    mode = args.data_config[channel_name]['TrainingInputMode']
    """Uses the tf.data input pipeline for our dataset.
    Args:
        mode: Standard names for model modes (tf.estimators.ModeKeys).
        batch_size: The number of samples per batch of input requested.
    """
    filenames = get_filenames(channel_name, channel)
    logging.info("Running {} in {} mode for {} epochs".format(channel_name, mode, epochs))

    # Repeat infinitely.
    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')
    else:
        dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(batch_size)

    # Parse records.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)
    ## TF Dataset question: why does _dataset_parser only get called once per channel??

    # Potentially shuffle records.
    if channel_name == 'train':
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        buffer_size = args.num_train_samples // args.batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    iterator = dataset.make_one_shot_iterator()
    features_batch, label_batch = iterator.get_next()
    
    with tf.Session() as sess:
#        logging.info('features_batch: {}'.format(features_batch.values))
        logging.info('type of features_batch: {}, type of values: {}'.format(type(features_batch), 
                                                         type(features_batch)))
        logging.info('label_batch: {}'.format(label_batch))
        logging.info('type of label_batch: {}'.format(type(label_batch)))

    return {INPUT_TENSOR_NAME: features_batch}, label_batch

def save_model(model, output):
    logging.info('Saving model, here are the contents:')
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, output)
    list_files_in_dir(output)
    logging.info('Model successfully saved at: {}'.format(output))
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_train_samples', type=int)
    parser.add_argument('--num_val_samples', type=int)
    parser.add_argument('--num_test_samples', type=int)
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--data-config', type=json.loads, 
                        default=os.environ.get('SM_INPUT_DATA_CONFIG'))

    args, _ = parser.parse_known_args()
    logging.info('args: {}'.format(args))
    epochs = args.epochs
    
    logging.info('getting data')
    train_dataset = train_input_fn()
    test_dataset  = test_input_fn()
    val_dataset   = val_input_fn()

    logging.info("configuring model")
    
    network = models.Sequential()
    network.add(layers.Dense(32, activation='relu', input_shape=(NUM_FEATURES,), name='inputs'))
    network.add(layers.Dense(32, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))
    network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    fitCallbacks = [ModelCheckpoint(os.environ.get('SM_OUTPUT_DATA_DIR') + '/checkpoint-{epoch}.h5')]
        #, PipeDebugCallback()]
    logging.info('Starting training')

    network.fit(x=train_dataset[0], y=train_dataset[1],
                steps_per_epoch=(args.num_train_samples // args.batch_size),
                epochs=args.epochs, 
                validation_data=val_dataset,
                validation_steps=(args.num_val_samples // args.batch_size),
                callbacks=fitCallbacks)

    logging.info('\nTraining completed.')
    logging.info('\nEvaluating against test set...')

    score = network.evaluate(test_dataset[0], test_dataset[1], 
                             steps=args.num_test_samples // args.batch_size,
                             verbose=1)
#    callbacks=[PipeDebugCallback()])

    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))

    logging.info('\nSaving model')
    save_model(network, os.environ.get('SM_MODEL_DIR'))
    logging.info('\nExiting training script.')