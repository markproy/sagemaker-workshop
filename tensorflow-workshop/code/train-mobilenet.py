# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
LAST_FROZEN_LAYER = 20

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

TF_VERSION = tf.version.VERSION
print('TF version: {}'.format(tf.__version__))
print('Keras version: {}'.format(tensorflow.keras.__version__))

import numpy as np
import os
import json
import argparse
import glob

HEIGHT = 224
WIDTH  = 224

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    # Freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) 
        if (dropout != 0.0):
            x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax', name='output')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


def create_data_generators(args):
    train_datagen =  ImageDataGenerator(
              preprocessing_function=preprocess_input,
              rotation_range=70,
              brightness_range=(0.6, 1.0),
              width_shift_range=0.3,
              height_shift_range=0.3,
              shear_range=0.3,
              zoom_range=0.3,
              horizontal_flip=True,
              vertical_flip=False)
    val_datagen  = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory('/opt/ml/input/data/train',
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=args.batch_size)
    test_gen = train_datagen.flow_from_directory('/opt/ml/input/data/test',
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=args.batch_size)
    val_gen = train_datagen.flow_from_directory('/opt/ml/input/data/validation',
                                                  target_size=(HEIGHT, WIDTH), 
                                                  batch_size=args.batch_size)
    return train_gen, test_gen, val_gen

def save_model_artifacts(model, model_dir):
    print(f'Saving model to {model_dir}...')
    # Note that this method of saving does produce a warning about not containing the train and evaluate graphs.
    # The resulting saved model works fine for inference. It will simply not support incremental training. If that
    # is needed, one can use model checkpoints and save those.
    print('Model directory files BEFORE save: {}'.format(glob.glob(f'{model_dir}/*/*')))
    if tf.version.VERSION[0] == '2':
        model.save(f'{model_dir}/1', save_format='tf')
    else:
        tf.contrib.saved_model.save_keras_model(model, f'{model_dir}/1')
    print('Model directory files AFTER save: {}'.format(glob.glob(f'{model_dir}/*/*')))
    print('...DONE saving model!')

    # Need to copy these files to the code directory, else the SageMaker endpoint will not use them.
    print('Copying inference source files...')

    if not os.path.exists(f'{model_dir}/code'):
        os.system(f'mkdir {model_dir}/code')
    os.system(f'cp inference.py {model_dir}/code')
    os.system(f'cp requirements.txt {model_dir}/code')
    print('Files after copying custom inference handler files: {}'.format(glob.glob(f'{model_dir}/code/*')))

def main(args):
    sm_training_env_json = json.loads(os.environ.get('SM_TRAINING_ENV'))
    is_master = sm_training_env_json['is_master']
    print('is_master {}'.format(is_master))
    
    # Create data generators for feeding training and evaluation based on data provided to us
    # by the SageMaker TensorFlow container
    train_gen, test_gen, val_gen = create_data_generators(args)

    base_model = MobileNetV2(weights='imagenet', 
                          include_top=False, 
                          input_shape=(HEIGHT, WIDTH, 3))

    # Here we extend the base model with additional fully connected layers, dropout for avoiding
    # overfitting to the training dataset, and a classification layer
    fully_connected_layers = []
    for i in range(args.num_fully_connected_layers):
        fully_connected_layers.append(1024)

    num_classes = len(glob.glob('/opt/ml/input/data/train/*'))
    model = build_finetune_model(base_model, 
                                  dropout=args.dropout, 
                                  fc_layers=fully_connected_layers, 
                                  num_classes=num_classes)

    opt = RMSprop(lr=args.initial_lr)
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    print('\nBeginning training...')
    
    NUM_EPOCHS  = args.fine_tuning_epochs
    FINE_TUNING = True
    
    num_train_images = len(train_gen.filepaths)
    num_val_images   = len(val_gen.filepaths)

    num_hosts   = len(args.hosts) 
    train_steps = num_train_images // args.batch_size // num_hosts
    val_steps   = num_val_images   // args.batch_size // num_hosts

    print('Batch size: {}, Train Steps: {}, Val Steps: {}'.format(args.batch_size, train_steps, val_steps))

    if not FINE_TUNING:
        history = model.fit_generator(train_gen, epochs=NUM_EPOCHS, workers=8, 
                               steps_per_epoch=train_steps, 
                               validation_data=val_gen, validation_steps=val_steps,
                               shuffle=True) 
    else:
        # Train for a few epochs
        model.fit_generator(train_gen, epochs=args.initial_epochs, workers=8, 
                               steps_per_epoch=train_steps, 
                               validation_data=val_gen, validation_steps=val_steps,
                               shuffle=True) 

        # Now fine tune the last set of layers in the model
        for layer in model.layers[LAST_FROZEN_LAYER:]:
            layer.trainable = True

        fine_tuning_lr = args.fine_tuning_lr
        model.compile(optimizer=SGD(lr=fine_tuning_lr, momentum=0.9, decay=fine_tuning_lr / NUM_EPOCHS), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit_generator(train_gen, epochs=NUM_EPOCHS, workers=8, 
                               steps_per_epoch=train_steps, 
                               validation_data=val_gen, validation_steps=val_steps,
                               shuffle=True)
    print('Model has been fit.')

    # Save the model if we are executing on the master host
    if is_master:
        print('Saving model, since we are master host')
        save_model_artifacts(model, os.environ.get('SM_MODEL_DIR'))
    else:
        print('NOT saving model, will leave that up to master host')

    print('\nExiting training script.\n')
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--initial_epochs', type=int, default=5)
    parser.add_argument('--fine_tuning_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--initial_lr', type=float, default=0.0001)
    parser.add_argument('--fine_tuning_lr', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_fully_connected_layers', type=int, default=1)
    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))

    args, _ = parser.parse_known_args()
    print('args: {}'.format(args))
    
    main(args)