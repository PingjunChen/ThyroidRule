# -*- coding: utf-8 -*-

import os, sys
import json, random
import numpy as np
import matplotlib.pyplot as plt

diag_mapping = {"Benign": 1, "Unsure": 2, "Malignant": 3}



def load_gt_pred(gt_json, pred_json):
    gt_dict = json.load(open(gt_json))
    pred_dict = json.load(open(pred_json))

    return gt_dict, pred_dict


def calculate_conf_mat(gt_dict, pred_dict):
    conf_mat = np.zeros((3, 3))
    for id in gt_dict:
        gt_diag = gt_dict[id]
        r_ind = int(gt_diag) - 1
        pred_diag = diag_mapping[pred_dict[id]]
        c_ind = int(pred_diag) - 1
        conf_mat[r_ind, c_ind] += 1

    return conf_mat


def sampling_conf_mat(gt_dict, pred_dict, s_ratio=0.8):
    gt_keys = list(gt_dict.keys())
    s_num = int(len(gt_keys) * s_ratio)
    sample_keys = random.sample(gt_keys, k=s_num)

    conf_mat = np.zeros((3, 3))
    for id in sample_keys:
        gt_diag = gt_dict[id]
        r_ind = int(gt_diag) - 1
        pred_diag = diag_mapping[pred_dict[id]]
        c_ind = int(pred_diag) - 1
        conf_mat[r_ind, c_ind] += 1

    return conf_mat


def metrics_conf_mat(conf_mat):
    benign_pre = conf_mat[0][0] * 1.0 / np.sum(conf_mat[:, 0])
    mal_pre = conf_mat[2][2] * 1.0 / np.sum(conf_mat[:, 2])
    un_sen = conf_mat[1][1] * 1.0 / np.sum(conf_mat[1, :])
    acc = np.trace(conf_mat) / np.sum(conf_mat)

    return benign_pre, mal_pre, un_sen, acc


def random_stats(conf_mat, select_ratio, select_num):
    benign_pre_list = []
    mal_pre_list = []
    un_sen_list = []
    acc_list = []
    for ind in range(select_num):
        sample_conf_mat = sampling_conf_mat(gt_dict, pred_dict, s_ratio=select_ratio)
        benign_pre, mal_pre, un_sen, acc = metrics_conf_mat(sample_conf_mat)
        benign_pre_list.append(benign_pre)
        mal_pre_list.append(mal_pre)
        un_sen_list.append(un_sen)
        acc_list.append(acc)

    return benign_pre_list, mal_pre_list, un_sen_list, acc_list


def draw_stat_boxplot(benign_pre_list, mal_pre_list, un_sen_list, acc_list):
    # fill with colors
    colors = ['pink', 'lightblue', 'lime', 'lightgreen']
    data_list = [benign_pre_list, mal_pre_list, un_sen_list, acc_list]
    fig, ax = plt.subplots(figsize=(9, 5))
    bplot = ax.boxplot(data_list, vert=True, patch_artist=True)   # fill with color
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(data_list))], )

    # add x-tick labels
    plt.setp(ax, xticks=[y+1 for y in range(len(data_list))],
             xticklabels=['Benign Precision', 'Malignant Precision', 'Uncertain Sensitivity', 'Overall Accuracy'])
    plt.title('Statistical Analysis of Key Evaluation Metrics')
    plt.savefig('eval_stat_analysis.pdf')
    # plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    gt_json = "./GTdiagnosis/gt_diagnosis.json"
    pred_json = "./Prediction/model-0.96-8-30.json" # prone to be unsure
    select_ratio, select_num = 0.8, 100

    gt_dict, pred_dict = load_gt_pred(gt_json, pred_json)
    conf_mat = calculate_conf_mat(gt_dict, pred_dict)
    # print("Confusion Matrix is:")
    # print(conf_mat)
    # sample_conf_mat = sampling_conf_mat(gt_dict, pred_dict, s_ratio=select_ratio)
    # # print("Sampling Confusion Matrix is:")
    # # print(sample_conf_mat)
    # mal_pre, un_sen, acc = metrics_conf_mat(sample_conf_mat)
    # print("Sampling malignant precision: {:.3f}".format(mal_pre))
    # print("Sampling uncertain sensitivity: {:.3f}".format(un_sen))
    # print("Overall accuracy: {:.3f}".format(acc))

    benign_pre_list, mal_pre_list, un_sen_list, acc_list = random_stats(conf_mat, select_ratio, select_num)
    # print("Sampling mean malignant precision: {:.3f}".format(np.mean(mal_pre_list)))
    # print("Sampling mean uncertain sensitivity: {:.3f}".format(np.mean(un_sen_list)))
    # print("Sampling mean accuracy: {:.3f}".format(np.mean(acc_list)))
    draw_stat_boxplot(benign_pre_list, mal_pre_list, un_sen_list, acc_list)
