# -*- coding: utf-8 -*-

import os, sys, pdb
import warnings
warnings.filterwarnings("ignore")
import json, argparse, time
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models
import torch.backends.cudnn as cudnn

from datetime import datetime
import shutil
from skimage import io, filters

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


def slide_predict(slide_path, cls_model, slide_save_dir, args):
    # print("Step 1: Split slide to patches")
    split_arr, patch_list, wsi_dim, s_img, mask = kfb_util.split_regions(
        slide_path, args.img_level, args.cnt_level)

    # print("Step 2: Predict Slide")
    pred_img, diag_flag = wsi_util.slide_pred(
        cls_model, split_arr, np.asarray(patch_list), wsi_dim, args)

    # # print("Step 3: Save results")
    # save_resize_ratio = 0.4
    # os.makedirs(slide_save_dir)
    # s_img = (s_img*255).astype(np.uint8)
    # io.imsave(os.path.join(slide_save_dir, "s_img.png"), misc.imresize(s_img, save_resize_ratio))
    # overlay_mask = img_util.mask_overlay_image(s_img, mask)
    # io.imsave(os.path.join(slide_save_dir, "overlay_mask.png"), misc.imresize(overlay_mask, save_resize_ratio))
    #
    # cmap = plt.get_cmap('jet')
    # resize_benign_map = misc.imresize(pred_img[:,:,0], save_resize_ratio)
    # io.imsave(os.path.join(slide_save_dir, "pred_benign.png"), cmap(resize_benign_map))
    # resize_unsure_map = misc.imresize(pred_img[:,:,1], save_resize_ratio)
    # io.imsave(os.path.join(slide_save_dir, "pred_unsure.png"), cmap(resize_unsure_map))
    # resize_malign_map = misc.imresize(pred_img[:,:,2], save_resize_ratio)
    # io.imsave(os.path.join(slide_save_dir, "pred_malign.png"), cmap(resize_malign_map))

    return diag_flag


# # special list for testing
# special_list = ["1238408", "1238690-1", ]

def predict_all_slides(model, args):
    kfb_list = [os.path.join(args.test_slide_dir, ele) for ele in os.listdir(args.test_slide_dir) if ele.endswith(".kfb")]
    print("There are {} kfb files in totoal.".format(len(kfb_list)))
    kfb_list.sort()

    print("Start processing...")
    print("="*80)
    slide_start = time.time()
    diag_dict = {}

    model_tag = args.model_name[:args.model_name.find("-")]
    for ind, kfb_filename in enumerate(kfb_list):
        slide_filename = os.path.splitext(os.path.basename(kfb_filename))[0]
        slide_save_dir = os.path.join(args.save_dir, model_tag, slide_filename)
        # if os.path.exists(slide_save_dir):
        #     continue
        # if slide_filename not in special_list:
        #     continue  # test for specfic slides

        # Get current slide information and print
        start_time = datetime.now()
        slide_img = kfbslide.open_kfbslide(kfb_filename)
        slide_width, slide_height = slide_img.level_dimensions[0]
        print("Processing {}, width: {}, height: {}, {}/{}".format(
            slide_filename, slide_width, slide_height, ind+1, len(kfb_list)))
        # Make prediction on this slide
        diag = slide_predict(kfb_filename, model, slide_save_dir, args)
        elapsed_time = datetime.now()-start_time
        print("Takes {}".format(elapsed_time))
        diag_dict[slide_filename] = diag
    print("="*80)
    slide_elapsed = time.time() - slide_start
    print("Time cost: " + time.strftime("%H:%M:%S", time.gmtime(slide_elapsed)))
    print("Finish Prediction...")

    save_json_path = os.path.join(args.save_dir, args.save_json_name + ".json")
    with open(save_json_path, "w") as outfile:
        json.dump(diag_dict, outfile)


def set_args():
    parser = argparse.ArgumentParser(description="Settings for thyroid slide prediction")
    parser.add_argument('--model_dir',            type=str, default="../data/TorchModels/ModelBest")
    parser.add_argument('--model_name',           type=str, default="Thyroid01-InceptionV3-0.9973.pth")
    parser.add_argument('--test_slide_dir',       type=str, default="../data/TestSlides")
    parser.add_argument('--save_dir',             type=str, default="../data/Results01")
    parser.add_argument('--save_json_name',       type=str, default="Model9973-0.98-8-30")
    parser.add_argument('--num_class',            type=int, default=3)
    parser.add_argument('--batch_size',           type=int, default=64)
    parser.add_argument('--img_level',            type=int, default=3)
    parser.add_argument('--cnt_level',            type=int, default=4)
    parser.add_argument('--malignant_prob',       type=float, default=0.98)
    parser.add_argument('--unsure_prob',          type=float, default=0.40)
    parser.add_argument('--unsure_grid_num',      type=int, default=36)
    parser.add_argument('--malignant_num_min',    type=int, default=8)
    parser.add_argument('--malignant_num_max',    type=int, default=30)
    parser.add_argument('--seed',                 type=int, default=1234)
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args = set_args()
    # Set prediction model
    args.model_path = os.path.join(args.model_dir, args.model_name)
    assert os.path.exists(args.model_path), "Model path does not exist"
    ft_model = load_model(args)
    print("Prediction model is: {}".format(args.model_name))

    # Predict All Patches
    predict_all_slides(model=ft_model, args=args)
