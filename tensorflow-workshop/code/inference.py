print('******* in inference.py *******')
import tensorflow as tf
print(f'TensorFlow version is: {tf.version.VERSION}')

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

import io
import json
import numpy as np
from collections import namedtuple
from PIL import Image

HEIGHT = 224
WIDTH  = 224

num_inferences = 0
print(f'num_inferences: {num_inferences}')

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')

def input_handler(data, context):

    global num_inferences
    num_inferences += 1
    
    print(f'\n*** in input_handler, inference #: {num_inferences}')
    if context.request_content_type == 'application/x-image':

        stream = io.BytesIO(data.read())
        img = Image.open(stream)
        img = img.convert('RGB')
        
        # Retrieve the attributes of the image
        fileFormat      = img.format       
        imageMode       = img.mode        
        imageSize       = img.size          # tuple of (width, height)
        colorPalette    = img.palette       
        
        print(f'    File format: {fileFormat}')
        print(f'    Image mode:  {imageMode}')
        print(f'    Image size:  {imageSize}')
        print(f'    Color pal:   {colorPalette}')
        
        print(f'    Keys from image.info dictionary:')
        for key, value in img.info.items():
            print(f'      {key}')
            
        img = img.resize((WIDTH, HEIGHT))
        x = np.asarray(img)
        curr_shape = x.shape
        new_shape = (1,) + curr_shape
        x = x.reshape(new_shape)
        instance = preprocess_input(x)
        del x, img
        print(f'    final image shape: {instance.shape}')
        inst_json = json.dumps({"instances": instance.tolist()})
        print(f'   returning from input_handler:\n')
        
        return inst_json

    else:
        _return_error(415, 'Unsupported content type "{}"'.format(context.request_content_type or 'Unknown'))

    return data

def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    print('*** in output_handler')

    if data.status_code != 200:
        raise Exception(data.content.decode('utf-8'))
    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))
