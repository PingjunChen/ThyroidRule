# -*- coding: utf-8 -*-

import os, sys, pdb
import shutil

from torchvision import models
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from visdom import Visdom
from loader import train_loader, val_loader


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (args.decay_ratio ** (epoch // args.lr_decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, epoch, train_acc, val_acc, win, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 2)).cpu() * epoch,
        Y=torch.Tensor([train_acc, val_acc]).unsqueeze(0).cpu() / epoch_size,
        win=win,
        update=update_type
    )


def train_thyroid(args):
    # construct model
    model = None
    if args.model_name == "InceptionV3":
        model = models.inception_v3(pretrained=args.pretrained)
        ## Change the last layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_class)
    elif args.model_name == "VGG16BN":
        model = models.vgg16_bn(pretrained=args.pretrained)
        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, args.num_class))
        model.classifier = nn.Sequential(*feature_model)
    elif args.model_name == "ResNet50":
        model = models.resnet50(pretrained=args.pretrained)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, args.num_class)
    else:
        raise Exception("Unknown model: {}".format(args.model_name))

    # cuda setting
    torch.cuda.manual_seed(args.seed)
    model.cuda()
    cudnn.benchmark = True

    # optimizer & loss
    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                       weight_decay=5.0e-4, momentum=0.9, nesterov=True)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99,
                              weight_decay=1.0e-6, momentum=0.9)
    criterion =nn.CrossEntropyLoss()
    # dataloader
    train_thyroid_loader = train_loader(args.batch_size)
    val_thyroid_loader = val_loader(args.batch_size)

    # folder for model saving
    if args.pretrained == True:
        model_save_dir = os.path.join(args.model_dir, "ft-"+args.model_name, args.session)
    else:
        model_save_dir = os.path.join(args.model_dir, "new-"+args.model_name, args.session)
    if os.path.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir)

    # viz = Visdom()
    # vis_title = "Thyroid classification"
    # vis_legend = ['Train Acc', 'Val Acc']
    # acc_plot = create_vis_plot(viz, 'Epoch', 'Acc', vis_title, vis_legend)
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train_acc = train(train_thyroid_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        val_acc = validate(val_thyroid_loader, model, criterion, args)
        is_best = val_acc > best_acc
        # update_vis_plot(viz, epoch, train_acc, val_acc, acc_plot, 'append')
        if is_best == True:
            best_acc = val_acc
            cur_model_name = str(epoch).zfill(2) + "-{:.3f}.pth".format(best_acc)
            torch.save(model.cpu(), os.path.join(model_save_dir, cur_model_name))
            print('Save weights at {}/{}'.format(model_save_dir, cur_model_name))
            model.cuda()


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()

    correct, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        if args.model_name == "InceptionV3":
            outputs, _ = model(inputs)
        elif args.model_name == "VGG16BN":
            outputs = model(inputs)
        elif args.model_name == "ResNet50":
            outputs = model(inputs)
        else:
            raise Exception("Unknown model: {}".format(args.model_name))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            batch_progress = 100. * batch_idx / len(train_loader)
            print("Train Epoch: {} [{}/{} ({:.2f})%)]\t Loss: {:.6f}".format(
                epoch, batch_idx, len(train_loader),
                batch_progress, loss.item()))
    train_acc = correct*1.0/total
    print("Training accuracy on Epoch {} is [{}/{} {:.4f})]".format(
        epoch, correct, total, train_acc))

    return train_acc


def validate(val_loader, model, criterion, args):
   model.eval()

   with torch.no_grad():
       correct, total = 0, 0
       for i, (inputs, targets) in enumerate(val_loader):
           inputs, targets = inputs.cuda(), targets.cuda()
           inputs, targets = Variable(inputs), Variable(targets)
           outputs = model(inputs)
           loss = criterion(outputs, targets)

           _, predicted = outputs.max(1)
           total += targets.size(0)
           correct += predicted.eq(targets).sum().item()

       val_acc = correct * 1.0 / total
       print("Validation accuracy is [{}/{} {:.4f})]".format(correct, total, val_acc))
   return val_acc
