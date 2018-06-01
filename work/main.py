#!/usr/bin/env python3

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import torchvision

import os
import sys
import time
import argparse
import collections

from torch.autograd import Variable

from utils import progress_bar
from dataset import VideoDataset

# Import models
import densenet
import resnext
import wide_resnet
from models import FeatRNN, FullModel, Pass
import transform

import tensorboard_logger

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resolution', default=224, type=int, help='image resolution')
parser.add_argument('--train_batch_size', default=32, type=int, help='batch size')
parser.add_argument('--test_batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='num workers')
parser.add_argument('--num_frames', default=16, type=int, help='num frames in time slice')
parser.add_argument('--train_slices', default=16, type=int, help='num frames in time slice')
parser.add_argument('--test_slices', default=16, type=int, help='num frames in time slice')
###
parser.add_argument('--tensor_dir', type=str, default='./rtdata/tensor')
parser.add_argument('--video_dir', type=str, default='./rtdata/video')
parser.add_argument('--storage', type=str, default='video')
parser.add_argument('--tag_path', type=str, 
        default='./movie-trailers-dataset/youtube_ids.txt')
parser.add_argument('--raw_path', type=str, 
        default='./movie-trailers-dataset/rt.json')
###
parser.add_argument('--mode', type=str, choices=['train', 'test', 'extract', 'norm', 'lstm', 'full', 'profile'], default='train')
parser.add_argument('--model_dir', type=str, default='./trained')
parser.add_argument('--feature_dir', type=str, default='./features')
parser.add_argument('--lstm_path', type=str, default='./checkpoint/epoch190.t7')
parser.add_argument('--dep_metadata_path', type=str, default='./metadata/dep.t7')
parser.add_argument('--modality', type=str, choices=['rgb', 'dep', 'all', 'fuse', 'pico'], default='rgb')
parser.add_argument('--load_modality', type=str, choices=['rgb', 'dep', 'all', 'fuse', 'pico'], default='rgb')
parser.add_argument('--method', type=str, choices=['class', 'auto'], default='class')
parser.add_argument('--shuffle', default=1, type=int, help='shuffle')
parser.add_argument('--preload', default=1, type=int, help='preload')
parser.add_argument('--load_epoch', default=None, type=int, help='which epoch to load from')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_loss = float('inf')  # best test accuracy
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

tensorboard_logger.configure('./logs/run14')

if True:
    vid_mean = (0.4338692858039216, 0.4045515923137255, 0.3776087500392157)
    vid_std = (0.1519876776470588, 0.14855877368627451, 0.15697639709803923)
else:
    vid_mean = (0.2108621746301651, 0.20068754255771637, 0.18471264839172363)
    vid_std =  (0.17436812818050385, 0.16858941316604614, 0.16036531329154968)

# Data
print('==> Preparing data..')
train_transform = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.Resize((args.resolution, args.resolution)),
    #transforms.ToTensor(),
    transform.TensorRead(),
    transforms.Normalize(vid_mean, vid_std),
    #transforms.ToPILImage()
])

test_transform = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.Resize((args.resolution, args.resolution)),
    #transforms.ToTensor(),
    transform.TensorRead(),
    transforms.Normalize(vid_mean, vid_std),
    #transforms.ToPILImage()
])


if args.mode == 'train':
    eval_mode = False
    train_slice_transform = transform.RandomScaledVideoCrop(args.resolution)
    test_slice_transform = transform.CenterVideoCrop(args.resolution)
elif args.mode == 'profile':
    eval_mode = False
    train_slice_transform = transform.CenterVideoCrop(args.resolution)
    test_slice_transform = transform.CenterVideoCrop(args.resolution)
else:
    eval_mode = True
    train_slice_transform = transform.CenterVideo5Crop(args.resolution)
    test_slice_transform = transform.CenterVideo5Crop(args.resolution)
    #train_slice_transform = transform.CenterVideoCrop(args.resolution)
    #test_slice_transform = transform.CenterVideoCrop(args.resolution)

# Set dataset type
DatasetType = VideoDataset

trainset = DatasetType(args.tensor_dir, args.video_dir, args.raw_path, 'train',
        transform=train_transform, slice_transform=train_slice_transform,
        num_frames=args.num_frames, eval_mode=eval_mode, num_slices=args.train_slices, preload=args.preload, storage=args.storage)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

valset = DatasetType(args.tensor_dir, args.video_dir, args.raw_path, 'val',
        transform=test_transform, slice_transform=test_slice_transform,
        num_frames=args.num_frames, eval_mode=eval_mode, num_slices=args.test_slices, preload=args.preload, storage=args.storage)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=args.shuffle, num_workers=args.num_workers)


print('==> Building model..')
if args.mode == 'train':
    #net = resnext.resnet101(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=2)
    #net = densenet.densenet121(sample_size=args.resolution, sample_duration=args.num_frames)#, num_classes=2)
    if args.load_epoch is None:
        #net = wide_resnet.resnet50(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=2)
        #net = densenet.densenet264(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=2)
        net = resnext.resnet101(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=400)
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        #state_dict = torch.load('./pretrained/resnext-101-64f-kinetics.pth')['state_dict']
        state_dict = torch.load('./pretrained/resnext-101-kinetics.pth')['state_dict']
        net.load_state_dict(state_dict)
        net.module.fc = torch.nn.Linear(2048, 2).cuda()
        #net = net.module
        #net.cuda()
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        #cudnn.benchmark = True
        #torch.backends.cudnn.enabled = False
        print('==> Loaded model')
    else:
        model_path  = os.path.join(args.model_dir, '{}_{}_{}.t7'.format(args.load_modality, args.method, args.load_epoch))
        checkpoint = torch.load(model_path)
        net = checkpoint['net']
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        loss = checkpoint['loss']
        print('--> Loading from epoch {}, acc={:.3f}, loss={:.3f}'.format(
            epoch, acc, loss))
        if use_cuda:
            cudnn.benchmark = True

elif args.mode == 'extract':
    model_path  = os.path.join(args.model_dir, '{}_{}_{}.t7'.format(args.load_modality, args.method, args.load_epoch))
    checkpoint = torch.load(model_path)
    net = checkpoint['net']
    classifier = net.fc
    net.fc = Pass()
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    loss = checkpoint['loss']
    print('--> Loading from epoch {}, acc={:.3f}, loss={:.3f}'.format(
        epoch, acc, loss))
    if use_cuda:
        classifier.cuda()
        classifier = torch.nn.DataParallel(classifier, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

elif args.mode == 'test' or args.mode == 'lstm':
    # Load spatial model
    model_path  = os.path.join(args.model_dir, '{}_{}_{}.t7'.format(args.load_modality, args.method, args.load_epoch))
    checkpoint = torch.load(model_path)
    net = checkpoint['net']
    classifier = net.fc
    net.fc = Pass()
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    loss = checkpoint['loss']
    print('--> Loading from epoch {}, acc={:.3f}, loss={:.3f}'.format(
        epoch, acc, loss))
    # Load temporal model
    #checkpoint = torch.load(args.lstm_path)
    #lstm = checkpoint['net']
    #feat_mean = checkpoint['mean']
    #feat_std = checkpoint['std']
    #acc = checkpoint['acc']
    #epoch = checkpoint['epoch']
    #print('==> Loading LSTM from epoch {}, acc={:2.3f}'.format(epoch, acc))
    if use_cuda:
        # Classifier
        classifier.cuda()
        classifier = torch.nn.DataParallel(classifier, device_ids=range(torch.cuda.device_count()))
        # LSTM
        #lstm.cuda()
        #lstm = torch.nn.DataParallel(lstm, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

elif args.mode == 'full':
    cnn_path  = os.path.join(args.model_dir, '{}_{}.t7'.format(args.load_modality, args.method))
    full_model = FullModel(cnn_path, args.lstm_path)

if False:#use_cuda and args.mode != 'profile':
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.mode != 'profile':
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

# Training
def profile(dataloader):
    print('\n==> Profiling..')
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        t1 = time.time()
        print('Time elapsed: {}'.format(t1-t0))
        t0 = t1

# Training
def train(net, epoch, dataloader):
    print('\n==> Training (epoch {})'.format(epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        targets = targets.squeeze().view((-1,))
        inputs = inputs.view((-1, args.num_frames, 3, args.resolution, args.resolution))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        try:
            outputs = net(inputs)
        except RuntimeError:
            print('Failed size:', inputs.size())
            continue
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        #print(targets)

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #if batch_idx > 150:
        #    break
    train_acc = correct/total
    train_loss = train_loss/len(dataloader)
    tensorboard_logger.log_value('train_acc', train_acc, epoch)
    tensorboard_logger.log_value('train_loss', train_loss, epoch)

def test(net, epoch, dataloader, save=True):
    print('\n==> Testing (epoch {})'.format(epoch))
    global best_loss
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        targets = targets.squeeze().view((-1,))
        inputs = inputs.view((-1, args.num_frames, 3, args.resolution, args.resolution))
        if use_cuda:
            try:
                inputs, targets = inputs.cuda(), targets.cuda()
                pass
            except RuntimeError:
                print('Failed cuda size:', inputs.size(), targets.size())
                continue
        inputs, targets = Variable(inputs), Variable(targets)
        try:
            with torch.no_grad():
                outputs = net(inputs)
        except RuntimeError:
            print('Failed net size:', inputs.size())
            raise
        with torch.no_grad():
            loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    test_loss = test_loss/len(dataloader)
    test_acc = correct/total
    tensorboard_logger.log_value('test_acc', test_acc, epoch)
    tensorboard_logger.log_value('test_loss', test_loss, epoch)

    # Save checkpoint.
    if save and (test_loss < best_loss or test_acc > best_acc):
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': test_acc,
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        model_path  = os.path.join(args.model_dir, '{}_{}_{}.t7'.format(args.modality, args.method, epoch))
        torch.save(state, model_path)
        if test_loss < best_loss:
            best_loss = test_loss
        if test_acc > best_acc:
            best_acc = test_acc

def extract(dataloader, mode, max_chunk=30):
    net.eval()
    classifier.eval()
    correct = 0
    total = 0
    emb_dict = {}
    emb_list = []
    label_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.view((inputs.size(1), args.num_frames, 3, 112, 112))
        #inputs = inputs.view((64, -1, 3, 112, 112)).transpose(0, 1)
        targets = targets.squeeze()
        if use_cuda:
            targets = targets.cuda()
        targets = Variable(targets, volatile=True)

        # Chunk the video slices for memory constraints
        #print('Input size:', inputs.size())
        chunk_emb, chunk_scores = [], []
        for chunk_idx in range(math.ceil(inputs.size(0)/max_chunk)):
            chunk = inputs[chunk_idx*max_chunk:chunk_idx*max_chunk+max_chunk].clone()
            #print('Chunk size:', chunk.size())
            if use_cuda:
                chunk = chunk.cuda()
            chunk = Variable(chunk, volatile=True)
            try:
                features = net(chunk)
            except (RuntimeError, ValueError) as e:
                print('\nFailed batch, size:', inputs.size())
                continue
            scores = classifier(features)
            chunk_emb.append(features.cpu().data.clone())
            chunk_scores.append(scores.cpu().data.clone())
        emb_list.append(torch.cat(chunk_emb, dim=0))
        outputs = torch.cat(chunk_scores, dim=0)

        label_list.append(targets.cpu().data.clone())
        # Get argmax of score
        outputs = torch.sum(outputs, dim=0)

        _, predicted = torch.max(outputs, 0)
        total += targets.size(0)
        correct += predicted.eq(targets.data.cpu()).cpu().sum().item()

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('{} acc:'.format(mode), 100.*correct/total)
    emb_dict['features'] = emb_list
    emb_dict['labels'] = label_list
    emb_dict['acc'] = correct/total
    if not os.path.isdir(args.feature_dir):
        os.mkdir(args.feature_dir)
    torch.save(emb_dict, os.path.join(args.feature_dir, '{}_{}.t7'.format(args.modality, mode)))


def simple_test(dataloader, max_chunk=3):
    net.eval()
    correct = 0
    total = 0
    emb_list = []
    label_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        if use_cuda:
            targets = targets.cuda()
        targets = Variable(targets, volatile=True)
        
        # Chunk the video slices for memory constraints
        chunk_emb, chunk_scores = [], []
        for chunk_idx in range(math.ceil(inputs.size(0)/max_chunk)):
            chunk = inputs[chunk_idx*max_chunk:chunk_idx*max_chunk+max_chunk].clone()
            if use_cuda:
                chunk = chunk.cuda()
            chunk = Variable(chunk, volatile=True)
            try:
                scores = net(chunk)
            except RuntimeError:
                print('\nFailed batch, size:', inputs.size())
                continue
            chunk_scores.append(scores.cpu().data.clone())

        outputs = torch.cat(chunk_scores, dim=0)
        label_list.append(targets.cpu().data.clone())

        # Get argmax of score
        print(outputs)
        outputs = torch.sum(outputs, dim=0)
        _, predicted = torch.max(outputs, 0)
        total += targets.size(0)
        correct += predicted.eq(targets.data.cpu()).cpu().sum()

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('acc:', 100.*correct/total)

def cnn_test(dataloader, max_chunk=3):
    net.eval()
    classifier.eval()
    correct = 0
    total = 0
    emb_list = []
    label_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        if use_cuda:
            targets = targets.cuda()
        targets = Variable(targets, volatile=True)
        
        print(inputs.size())

        # Chunk the video slices for memory constraints
        chunk_emb, chunk_scores = [], []
        for chunk_idx in range(math.ceil(inputs.size(0)/max_chunk)):
            chunk = inputs[chunk_idx*max_chunk:chunk_idx*max_chunk+max_chunk].clone()
            if use_cuda:
                chunk = chunk.cuda()
            chunk = Variable(chunk, volatile=True)
            try:
                features = net(chunk)
            except RuntimeError:
                print('\nFailed batch, size:', inputs.size())
                continue
            print(features.size())
            scores = classifier(features)
            chunk_emb.append(features.cpu().data.clone())
            chunk_scores.append(scores.cpu().data.clone())
        emb_list.append(torch.cat(chunk_emb, dim=0))

        outputs = torch.cat(chunk_scores, dim=0)
        label_list.append(targets.cpu().data.clone())

        # Get argmax of score
        outputs = torch.sum(outputs, dim=0)
        print(outputs.size())
        _, predicted = torch.max(outputs, 0)
        total += targets.size(0)
        correct += predicted.eq(targets.data.cpu()).cpu().sum()

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('acc:', 100.*correct/total)

def lstm_test(dataloader, max_chunk=100):
    net.eval()
    classifier.eval()
    lstm.eval()
    correct = 0
    total = 0
    emb_list = []
    label_list = []
    conf_dict = collections.defaultdict(int)
    tot_dict = collections.defaultdict(int)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        if use_cuda:
            targets = targets.cuda()
        targets = Variable(targets, volatile=True)

        # Chunk the video slices for memory constraints
        #print('Input size:', inputs.size())
        chunk_emb, chunk_scores = [], []
        for chunk_idx in range(math.ceil(inputs.size(0)/max_chunk)):
            chunk = inputs[chunk_idx*max_chunk:chunk_idx*max_chunk+max_chunk].clone()
            #print('Chunk size:', chunk.size())
            if use_cuda:
                chunk = chunk.cuda()
            chunk = Variable(chunk, volatile=True)
            try:
                features = net(chunk)
            except RuntimeError:
                print('\nFailed batch, size:', inputs.size())
                continue
            scores = classifier(features)
            chunk_emb.append(features.cpu().data.clone())
            chunk_scores.append(scores.cpu().data.clone())
        emb = torch.cat(chunk_emb, dim=0)
        emb_list.append(emb)
        outputs = torch.cat(chunk_scores, dim=0)
        label_list.append(targets.cpu().data.clone())

        # Put output through LSTM
        inputs = emb
        inputs = (inputs - feat_mean) / feat_std
        if use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        inputs = inputs.unsqueeze(0)
        outputs = lstm(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        conf_dict[targets.cpu().data.numpy()[0]] += predicted.eq(targets.data).cpu().sum()
        tot_dict[targets.cpu().data.numpy()[0]] += targets.size(0)

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('acc:', 100.*correct/total)

    print('Class breakdown:')
    for c in tot_dict:
        print('{}: {:.2f}'.format(c, conf_dict[c]/tot_dict[c]))

def full_test(dataloader, max_chunk=100):
    full_model.eval()
    correct = 0
    total = 0
    emb_list = []
    label_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        for i, x in enumerate(inputs):
            print(x.size())
            sys.exit()
            pred = full_model(x)
            if pred == targets[i]:
                correct += 1
            total += 1

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('acc:', 100.*correct/total)

if __name__=='__main__':
    if args.mode == 'train':
        if args.load_epoch is not None:
            start_epoch = args.load_epoch
        for epoch in range(start_epoch, start_epoch+10000):
            train(net, epoch, trainloader)
            if epoch % 5 == 0:
                test(net, epoch, valloader)
    elif args.mode == 'test':
        simple_test(valloader)
        #cnn_test(valloader)
        #for i in range(5):
        #    test(epoch, valloader, save=False)
    elif args.mode == 'lstm':
        lstm_test(valloader)
    elif args.mode == 'full':
        full_test(valloader)
    elif args.mode == 'extract':
        extract(valloader, 'val')
        extract(trainloader, 'train')
    elif args.mode == 'profile':
        profile(valloader)
