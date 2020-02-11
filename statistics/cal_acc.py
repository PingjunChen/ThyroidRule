# @Date:   2018-05-29T12:16:36+08:00
# @Last modified time: 2018-08-14T02:53:17+08:00

import os, sys, pdb
import json
import numpy as np

diag_mapping = {"Benign": 1, "Unsure": 2, "Malignant": 3}

def cal_confusion_acc(conf_mat):
    sum_all = np.sum(conf_mat)
    sum_diag = np.trace(conf_mat)

    conf_acc = sum_diag * 1.0 / sum_all

    return conf_acc

def predict_confusion(gt_json, auto_json):
    gt_dict = json.load(open(gt_json))
    auto_dict = json.load(open(auto_json))

    conf_mat = np.zeros((3, 3))
    for id in gt_dict:
        gt_diag = gt_dict[id]
        r_ind = int(gt_diag) - 1
        auto_diag = diag_mapping[auto_dict[id]]
        c_ind = int(auto_diag) - 1
        conf_mat[r_ind, c_ind] += 1

    cls_acc = cal_confusion_acc(conf_mat)
    assert np.sum(conf_mat) == 259, "Total number of testing slides doese not match"

    return conf_mat, cls_acc

if __name__ == "__main__":
    gt_json = "./GTdiagnosis/gt_diagnosis.json"
    auto_json = "./Prediction/model-0.96-8-30.json" # prone to be unsure

    conf_mat, cls_acc = predict_confusion(gt_json, auto_json)
    print("Confusion Matrix is: \n {}".format(conf_mat))
    print("Classification accuracy is：{0:.3f}".format(cls_acc))
