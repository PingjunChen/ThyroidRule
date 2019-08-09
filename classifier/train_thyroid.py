# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import argparse
from train_eng import train_thyroid


def set_args():
    parser = argparse.ArgumentParser(description='Thyroid Classification')
    parser.add_argument('--num_class',       type=int,   default=3)
    parser.add_argument('--epochs',          type=int,   default=12)
    parser.add_argument('--batch_size',      type=int,   default=32)
    # Optimization parameters
    parser.add_argument('--lr',              type=float, default=1.0e-3)
    parser.add_argument('--decay_ratio',     type=float, default=0.8)
    parser.add_argument('--lr_decay_epochs', type=int,   default=2)
    parser.add_argument('--log_interval',    type=int,   default=200)
    # model directory and name
    parser.add_argument('--seed',            type=int,   default=1234)
    parser.add_argument('--model_dir',       type=str,   default="../data/PatchModels")
    parser.add_argument('--pretrained',      type=bool,  default=True) # True, False
    parser.add_argument('--gpu_id',          type=str,   default="3")
    parser.add_argument('--model_name',      type=str,   default="InceptionV3")
                        # options=["InceptionV3", "VGG16BN", "ResNet50"]
    parser.add_argument('--session',         type=str,   default="sess01")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    # np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # start training
    print("Starting training: {}".format(args.model_name))
    train_thyroid(args)
