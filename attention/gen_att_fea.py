# -*- coding: utf-8 -*-

import os, sys
import warnings
warnings.filterwarnings("ignore")
import argparse, time, shutil
from datetime import datetime
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import io, filters
import deepdish as dd

import torch
import torch.nn as nn
from torchvision import models
import torch.backends.cudnn as cudnn

# Add kfb support
FileAbsPath = os.path.abspath(__file__)
ProjectPath = os.path.dirname(os.path.dirname(FileAbsPath))
sys.path.append(os.path.join(ProjectPath, 'utils'))
import kfb_util, wsi_util, img_util
sys.path.append(os.path.join(ProjectPath, 'kfb'))
import kfbslide


def load_model(args):
    model = torch.load(args.model_path)
    torch.cuda.manual_seed(args.seed)
    model.cuda()
    cudnn.benchmark = True
    model.eval()

    return model


def predict_slide_fea(slide_path, cls_model, save_dir, args):
    file_fullname = os.path.basename(slide_path)
    file_name = file_fullname[:-4]
    fea_filepath = os.path.join(save_dir, file_name + ".h5")
    if os.path.exists(fea_filepath):
        return

    # print("Step 1: Split slide to patches")
    split_arr, patch_list, wsi_dim, s_img, mask = kfb_util.split_regions(
        slide_path, args.img_level, args.cnt_level)

    # print("Step 2: Generate features")
    fea_dict = wsi_util.gen_slide_feas(cls_model, split_arr, np.asarray(patch_list), wsi_dim, args)

    # save features
    dd.io.save(fea_filepath, fea_dict)

    # print("Step 4: Save image and mask to check the preprocessing")
    save_resize_ratio = 0.4
    s_img = (s_img*255).astype(np.uint8)
    io.imsave(os.path.join(save_dir, "-".join([file_name, "img.png"])), misc.imresize(s_img, save_resize_ratio))
    overlay_mask = img_util.mask_overlay_image(s_img, mask)
    io.imsave(os.path.join(save_dir, "-".join([file_name, "mask.png"])), misc.imresize(overlay_mask, save_resize_ratio))


def predit_all_feas(model, args):
    kfb_list = [os.path.join(args.slide_dir, ele) for ele in os.listdir(args.slide_dir) if ele.endswith(".kfb")]
    print("There are {} kfb files in totoal.".format(len(kfb_list)))
    kfb_list.sort()

    print("Start processing...")
    print("="*80)
    slide_start = time.time()

    model_tag = args.model_name[:args.model_name.find("-")]
    for ind, kfb_filename in enumerate(kfb_list):
        slide_filename = os.path.splitext(os.path.basename(kfb_filename))[0]

        # Get current slide information and print
        start_time = datetime.now()
        slide_img = kfbslide.open_kfbslide(kfb_filename)
        slide_width, slide_height = slide_img.level_dimensions[0]
        print("Processing {}, width: {}, height: {}, {}/{}".format(
            slide_filename, slide_width, slide_height, ind+1, len(kfb_list)))
        # Make prediction on this slide
        flag = predict_slide_fea(kfb_filename, model, args.fea_dir, args)
        elapsed_time = datetime.now()-start_time
        print("Takes {}".format(elapsed_time))
    print("="*80)
    slide_elapsed = time.time() - slide_start
    print("Time cost: " + time.strftime("%H:%M:%S", time.gmtime(slide_elapsed)))
    print("Finish Prediction...")


def set_args():
    parser = argparse.ArgumentParser(description="Settings for thyroid slide prediction")
    parser.add_argument('--model_dir',            type=str, default="../data/TorchModels/ModelBest")
    parser.add_argument('--model_name',           type=str, default="Thyroid01-InceptionV3-0.9973.pth")
    parser.add_argument('--slide_dir',            type=str, default="../data/TrainSlides")
    parser.add_argument('--fea_dir',              type=str, default="../data/Feas/TrainFeas")
    parser.add_argument('--num_class',            type=int, default=3)
    parser.add_argument('--batch_size',           type=int, default=64)
    parser.add_argument('--img_level',            type=int, default=3)
    parser.add_argument('--cnt_level',            type=int, default=4)
    parser.add_argument('--seed',                 type=int, default=1234)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    args = set_args()
    # Set prediction model
    args.model_path = os.path.join(args.model_dir, args.model_name)
    assert os.path.exists(args.model_path), "Model path does not exist"
    ft_model = load_model(args)
    print("Prediction model is: {}".format(args.model_name))

    # Predict All Patches
    predit_all_feas(model=ft_model, args=args)
