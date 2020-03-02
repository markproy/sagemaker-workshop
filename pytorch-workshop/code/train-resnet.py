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

import numpy as np
from numpy import argmax
import os
import io
import json
import argparse
import glob
import time
from PIL import Image

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import RandomSampler
import torch.utils.data.distributed
from torchvision.datasets import ImageFolder

HEIGHT = 224
WIDTH  = 224

INFERENCE_DROPOUT = 0.3
NUM_CLASSES = 4 #26 # 200 #13 #4

print(f'PyTorch version: {torch.__version__}')
top_time = time.time()
print(f'Time at entry to top of training script: {top_time}')
print(f'\nNumber of classes for this model: {NUM_CLASSES}\n')

def get_loaders(args, is_distributed, **kwargs):
    train_dataset = ImageFolder(root=args.train, 
                            transform=transforms.Compose([
                                       transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                                       transforms.RandomRotation(degrees=15),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.CenterCrop(size=224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std= [0.229, 0.224, 0.225])]))
    val_dataset = ImageFolder(root=args.validation,
                               transform=transforms.Compose([
                                   transforms.Resize(size=256),
                                   transforms.CenterCrop(size=224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std= [0.229, 0.224, 0.225])]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_distributed else None
    _trainloader  = torch.utils.data.DataLoader(train_dataset,
                                                sampler=train_sampler, 
                                                shuffle=train_sampler is None, 
                                                batch_size=args.batch_size, **kwargs)

    val_sampler = RandomSampler(val_dataset)
    _valloader  = torch.utils.data.DataLoader(val_dataset,
                                              sampler=val_sampler, 
                                              batch_size=args.batch_size, **kwargs)
    
    return _trainloader, _valloader

def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def main(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    print('Distributed training - {}'.format(is_distributed))

    use_cuda = args.num_gpus > 0
    print('Number of gpus available - {}'.format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')

    world_size = len(args.hosts)
    host_rank  = args.hosts.index(args.current_host)
    if is_distributed:
        # Initialize the distributed environment.
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        print('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))
    else:
        print(f'Host rank is {host_rank}. Number of gpus: {args.num_gpus}') 
        
    trainloader, valloader = get_loaders(args, is_distributed, **kwargs)

    print('Will process {}/{} ({:.0f}%) of train data'.format(
        len(trainloader.sampler), len(trainloader.dataset),
        100. * len(trainloader.sampler) / len(trainloader.dataset)))
    print('Will process {}/{} ({:.0f}%) of test data'.format(
        len(valloader.sampler), len(valloader.dataset),
        100. * len(valloader.sampler) / len(valloader.dataset)))

    pre_model_time = time.time()
    model = models.resnet50(pretrained=True, progress=False) 

    for param in model.parameters():
        param.requires_grad = False

    fc_inputs = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(fc_inputs, 256),
                                     nn.ReLU(),
                                     nn.Dropout(args.dropout),
                                     nn.Linear(256, len(trainloader.dataset.classes)),
                                     nn.LogSoftmax(dim=1)) # for using NLLLoss()
    model.to(device)
    post_model_time = time.time()
    print(f'\nTime to download resnet and replace final layers: {post_model_time - pre_model_time} seconds.')
    
    optimizer = optim.Adam(model.fc.parameters(), lr=args.initial_lr)
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    criterion = nn.NLLLoss()

    epochs = args.initial_epochs
    steps  = 0
    running_loss = 0
    print_every  = 3
    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss  = criterion(logps, labels)

            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy  = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                test_accuracies.append(accuracy/len(valloader))
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(valloader))                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(valloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(valloader):.3f}")
                running_loss = 0
                model.train()
                
    print('\nCompleted training.\n')

    if host_rank == 0:
        model_dir = args.model_dir
        path = f'{model_dir}/model.pth'
        print(f'Saving model to {path}...')
        torch.save(model.state_dict(), path) # model.cpu()
        print('Model directory files AFTER save: {}'.format(glob.glob(f'{model_dir}/*')))

    print('\nExiting training script.')

def input_fn(request_body, request_content_type):
    print('In input_fn...')
    stream = io.BytesIO(request_body)
    img = Image.open(stream)
    test_transforms = transforms.Compose([
                               transforms.Resize(size=256),
                               transforms.CenterCrop(size=224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std= [0.229, 0.224, 0.225])])
    image_tensor = test_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    
    input = Variable(image_tensor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = input.to(device)
    return input

def predict_fn(input_object, model):
    print('In predict_fn...')

    start_time = time.time()
    output_obj = model(input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))

    ps_torch = torch.exp(output_obj).detach().cpu()
    ps = ps_torch.numpy()[0]
    return ps.tolist() 

# Serialize the prediction result into the desired response content type
def output_fn(prediction, response_content_type):
    print('In output_fn...')
    return prediction

def model_fn(model_dir):
    print('In model_fn...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'===device is: {device}')

    model     = models.resnet50(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc  = nn.Sequential(nn.Linear(fc_inputs, 256),
                                     nn.ReLU(),
                                     nn.Dropout(INFERENCE_DROPOUT),
                                     nn.Linear(256, NUM_CLASSES),
                                     nn.LogSoftmax(dim=1)) # for using NLLLoss()
    print('Loaded pretrained resnet and updated fc layer')

    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))
    print('Updated model state dict from model.pth')

    model.to(device)
    print('===After model.to(device)')
    model.eval()
    print('===After model.eval()')
    
    if torch.cuda.device_count() > 1:
        print('GPU count: {}'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    
    return model


if __name__=='__main__':
    main_time = time.time()
    print(f'Time at __main__: {main_time}, lost time: {main_time - top_time} seconds.')

    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--initial_epochs', type=int, default=5)
    parser.add_argument('--fine_tuning_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--initial_lr', type=float, default=0.0001)
    parser.add_argument('--fine_tuning_lr', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_fully_connected_layers', type=int, default=1)
    
    # input data directories and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    # infra paramaters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    args, _ = parser.parse_known_args()
    print('args: {}'.format(args))
    
    num_train = len(glob.glob(f'{args.train}/*/*'))
    num_test  = len(glob.glob(f'{args.test}/*/*'))
    print(f'\nNumber of images already in training channel at start: {num_train}')
    print(f'\nNumber of images already in test     channel at start: {num_test}')
    
    main(args)