# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cls_p = {
    'ft-InceptionV3':  [0.792, 0.797, 0.792, 0.797, 0.791],
    'new-InceptionV3': [0.774, 0.782, 0.766, 0.779, 0.769],
    'ft-VGG16BN':      [0.785, 0.790, 0.779, 0.788, 0.787],
    'new-VGG16BN':     [0.744, 0.763, 0.742, 0.745, 0.752],
    'ft-ResNet50':     [0.787, 0.786, 0.794, 0.774, 0.784],
    'new-ResNet50':    [0.744, 0.751, 0.741, 0.737, 0.753]
}


model_manners = ['Fine-tuning', 'Training from scratch']

cls_dict = {
    'model_name': ['InceptionV3', 'VGG16BN', 'ResNet50'],
    'ft_mean':  [np.mean(cls_p['ft-InceptionV3']), np.mean(cls_p['ft-VGG16BN']), np.mean(cls_p['ft-ResNet50'])],
    'new_mean': [np.mean(cls_p['new-InceptionV3']), np.mean(cls_p['new-VGG16BN']), np.mean(cls_p['new-ResNet50'])],
    'ft_std':   [np.std(cls_p['ft-InceptionV3']), np.std(cls_p['ft-VGG16BN']), np.std(cls_p['ft-ResNet50'])],
    'new_std':  [np.std(cls_p['new-InceptionV3']), np.std(cls_p['new-VGG16BN']), np.std(cls_p['new-ResNet50'])],
}

df = pd.DataFrame(cls_dict, columns = ['model_name', 'ft_mean', 'new_mean', 'ft_std', 'new_std'])


pos = list(range(len(df['model_name'])))
width = 0.3

# Plotting the bars
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_ylim(ymin=0.72, ymax=0.82)

plt.bar(pos, height=df['ft_mean'], yerr=df['ft_std'], width=width, alpha=0.8, color='#1F76B4', ecolor='black', capsize=6)
plt.bar([p + width for p in pos], height=df['new_mean'], yerr=df['new_std'], width=width, alpha=0.8, color='#FF7F0F', ecolor='black', capsize=6)

# Set the y axis label
ax.set_ylabel('Accuracy')

# Set the chart's title
ax.set_title('Classification Accuracy Comparison')

# Set the position of the x ticks
ax.set_xticks([p + width-0.15 for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['model_name'])

plt.legend(model_manners, loc='upper left', fontsize='large')
plt.tight_layout()
# plt.grid()
# plt.show()
plt.savefig('classifier_cmp.pdf')
