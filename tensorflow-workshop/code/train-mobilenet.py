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

print('TF version: {}'.format(tf.__version__))
print('Keras version: {}'.format(tensorflow.keras.__version__))

import numpy as np
from numpy import argmax
import os
import argparse
import glob

HEIGHT = 224
WIDTH  = 224

# Need to copy these files to the code directory, else the SageMaker endpoint will not use them.
print('Copying inference source files...')

if not os.path.exists('/opt/ml/model/code'):
    os.system('mkdir /opt/ml/model/code')
os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')

print('Files after copy:')
print(glob.glob('/opt/ml/model/code/*'))

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


def create_data_generators():
    train_datagen =  ImageDataGenerator(
              preprocessing_function=preprocess_input,
              rotation_range=70,
              brightness_range=(0.7, 1.0),
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=0.2,
              zoom_range=0.2,
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
    
def main(args):
    # Create data generators for feeding training and evaluation based on data provided to us
    # by the SageMaker TensorFlow container
    train_gen, test_gen, val_gen = create_data_generators()

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

    if not FINE_TUNING:
        history = model.fit_generator(train_gen, epochs=NUM_EPOCHS, workers=8, 
                               steps_per_epoch=num_train_images // args.batch_size, 
                               validation_data=val_gen, validation_steps=num_val_images // args.batch_size,
                               shuffle=True) 
    else:
        # Train for a few epochs
        model.fit_generator(train_gen, epochs=args.initial_epochs, workers=8, 
                               steps_per_epoch=num_train_images // args.batch_size, 
                               validation_data=val_gen, validation_steps=num_val_images // args.batch_size,
                               shuffle=True) 

        # Now fine tune the last set of layers in the model
        for layer in model.layers[LAST_FROZEN_LAYER:]:
            layer.trainable = True

        fine_tuning_lr = args.fine_tuning_lr
        model.compile(optimizer=SGD(lr=fine_tuning_lr, momentum=0.9, decay=fine_tuning_lr / NUM_EPOCHS), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit_generator(train_gen, epochs=NUM_EPOCHS, workers=8, 
                               steps_per_epoch=num_train_images // args.batch_size, 
                               validation_data=val_gen, validation_steps=num_val_images // args.batch_size,
                               shuffle=True)
    print('Model has been fit.')

    print('Saving model to /opt/ml/model...')
    # Note that this method of saving does produce a warning about not containing the train and evaluate graphs.
    # The resulting saved model works fine for inference. It will simply not support incremental training. If that
    # is needed, one can use model checkpoints and save those.
    tf.contrib.saved_model.save_keras_model(model, '/opt/ml/model')
    print('...DONE saving model!')

    print('\nExiting training script.\n')
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--initial_epochs', type=int, default=5)
    parser.add_argument('--fine_tuning_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--initial_lr', type=float, default=0.0001)
    parser.add_argument('--fine_tuning_lr', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_fully_connected_layers', type=int, default=1)
    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args, _ = parser.parse_known_args()
    print('args: {}'.format(args))
    
    main(args)