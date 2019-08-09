# -*- coding: utf-8 -*-

import os, sys, pdb

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms

import numpy as np
import cv2, copy, time
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_closing, binary_dilation
from skimage import transform, morphology, filters
from skimage.morphology import remove_small_objects

import loader


def refine_prediction(pred, thresh, min_size):
    binary = pred > thresh  # Threshold
    binary = binary_dilation(binary, structure=np.ones((5,5))) # dilation to connect
    binary = binary_fill_holes(binary)  # Fill holes
    # Remove outliers
    mask = remove_small_objects(binary, min_size=min_size, connectivity=8)

    return mask


def pred_patches(cls_model, patches, args):
    preds = []

    start_time = time.time()
    slide_dset = loader.PatchDataset(patches)
    dset_loader = data.DataLoader(slide_dset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for ind, inputs in enumerate(dset_loader):
            inputs = inputs.type(torch.FloatTensor)
            inputs = Variable(inputs.cuda())
            outputs = cls_model(inputs)
            _, batch_preds = outputs.max(1)
            preds.extend(batch_preds.cpu().tolist())

    elapsed_time = time.time() - start_time
    print("{} seconds for {} patches.".format(elapsed_time, patches.shape[0]))
    
    return preds


def slide_pred(cls_model, split_arr, patches, wsi_dim, args):
    # Save prediction results
    RAW_SIZE = 299
    SIZE1, SIZE2, SIZE4 = int(RAW_SIZE/4), int(RAW_SIZE/2), RAW_SIZE
    class_num = 3
    result_map = np.zeros((wsi_dim[0], wsi_dim[1], class_num), dtype=np.uint8)

    # Prediction
    if patches.shape[0] > 0: # exist
        preds = pred_patches(cls_model, patches, args)
        for coor, pred in zip(split_arr, preds):
            result_map[coor[0]+SIZE1:coor[0]+SIZE1+SIZE2, coor[1]+SIZE1:coor[1]+SIZE1+SIZE2, pred] = 255

    # Resize results
    args.img_cnt_ratio = 2**(args.cnt_level - args.img_level)
    s_height, s_width = wsi_dim[0] / args.img_cnt_ratio, wsi_dim[1] / args.img_cnt_ratio
    result_img = transform.resize(result_map, (s_height, s_width))

    MINIMUM_REGION_SIZE = (np.floor(SIZE2 / args.img_cnt_ratio))**2
    # refine unsure
    unsure_min_size = MINIMUM_REGION_SIZE * args.unsure_grid_num
    result_img[:,:,1] = refine_prediction(result_img[:,:,1], thresh=args.unsure_prob, min_size=unsure_min_size)
    unsure_img = (result_img[:,:,1] * 255).astype(np.uint8)
    _, unsure_cnts, _ = cv2.findContours(unsure_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_unsure = 0
    if len(unsure_cnts) != 0:
        max_unsure_cnt = max(unsure_cnts, key = cv2.contourArea)
        max_unsure = cv2.contourArea(max_unsure_cnt)
    unsure_num_grid = int(max_unsure / MINIMUM_REGION_SIZE)
    # refine malignant
    yes_min_size = MINIMUM_REGION_SIZE * args.malignant_num_min
    result_img[:,:,2] = refine_prediction(result_img[:,:,2], thresh=args.malignant_prob, min_size=yes_min_size)
    yes_img = (result_img[:,:,2] * 255).astype(np.uint8)
    _, yes_cnts, _ = cv2.findContours(yes_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_yes = 0
    if len(yes_cnts) != 0:
        max_yes_cnt = max(yes_cnts, key = cv2.contourArea)
        max_yes = cv2.contourArea(max_yes_cnt)
    yes_num_grid = int(max_yes / MINIMUM_REGION_SIZE)

    # Rule-based diagnosis
    diag_flag = thyroid_diagnosis_rule(unsure_num_grid, yes_num_grid, args)
    return result_img, diag_flag


def thyroid_diagnosis_rule(unsure_num, yes_num, args):
    diag_flag = "Benign"
    # if there are unsure regions, take it unsure
    if unsure_num != 0:
        diag_flag = "Unsure"
    else:
        # if malignant regions large than 16, take it as malignant
        if yes_num >= args.malignant_num_max:
            diag_flag = "Malignant"
        # if malignant regions num between 2-16, take is as Unsure
        elif yes_num >= args.malignant_num_min and yes_num < args.malignant_num_max:
            diag_flag = "Unsure"
        else:
            diag_flag = "Benign"
    return diag_flag



def pred_feas(cls_model, patches, args):
    probs, logits, vecs = [], [], []

    def fea_hook(module, input, output):
        t_fea2048 = input[0].cpu().tolist()
        cur_vecs = copy.deepcopy(t_fea2048)
        t_logit3 = output.cpu().tolist()
        cur_logits = copy.deepcopy(t_logit3)
        t_fea3 = F.softmax(output, dim=-1)
        cur_fea3 = t_fea3.cpu().tolist()
        cur_probs = copy.deepcopy(cur_fea3)

        vecs.extend(cur_vecs)
        logits.extend(cur_logits)
        probs.extend(cur_probs)

    cls_model.fc.register_forward_hook(fea_hook)
    slide_dset = loader.PatchDataset(patches)
    dset_loader = data.DataLoader(slide_dset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for ind, inputs in enumerate(dset_loader):
            inputs = inputs.type(torch.FloatTensor)
            inputs = Variable(inputs.cuda())
            outputs = cls_model(inputs)

    return probs, logits, vecs



def sort_by_prob(BBoxes, ClsProbs, ClsLogits, FeaVecs):
    fea_dict = {}
    norm_prob_list = [ele[0] for ele in ClsProbs]
    sorting_indx = np.argsort(norm_prob_list)
    fea_dict["bbox"] = [BBoxes[ind] for ind in sorting_indx]
    fea_dict["prob"] = [ClsProbs[ind] for ind in sorting_indx]
    fea_dict["logit"] = [ClsLogits[ind] for ind in sorting_indx]
    fea_dict["feaVec"] = [FeaVecs[ind] for ind in sorting_indx]

    return fea_dict


def gen_slide_feas(cls_model, split_arr, patches, wsi_dim, args):
    RAW_SIZE = 299
    SIZE1, SIZE2, SIZE4 = int(RAW_SIZE/4), int(RAW_SIZE/2), RAW_SIZE
    class_num = 3

    FeasList = []
    BBoxes, ClsProbs, ClsLogits, FeaVecs = [], [], [], []
    # Prediction
    if patches.shape[0] > 0: # exist
        ClsProbs, ClsLogits, FeaVecs = pred_feas(cls_model, patches, args)
        for coor in split_arr:
            cur_x, cur_y = coor[1]+SIZE1, coor[0]+SIZE1
            cur_bbox = [cur_x, cur_y, SIZE2, SIZE2]
            BBoxes.append(cur_bbox)

    fea_dict = sort_by_prob(BBoxes, ClsProbs, ClsLogits, FeaVecs)
    return fea_dict
