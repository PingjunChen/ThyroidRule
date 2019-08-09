import os, sys, pdb
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage import filters
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects


def mask_overlay_image(img, mask, alpha=0.5):
    assert img.shape[:2] == mask.shape[:2], "Image and Mask shape not match"
    if np.amax(img) > 1:
        img = img / 255.0
    if np.amax(mask) > 1:
        mask = mask / 255.0

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask_rgb = mask
    else:
        zero_img = np.zeros((img.shape[0], img.shape[1]))
        # mask_rgb = np.stack((zero_img, zero_img, mask), axis=2)
        mask_rgb = np.stack((mask, zero_img, zero_img), axis=2)
    overlay_img = cv2.addWeighted(mask_rgb, alpha, img, 1 - alpha, 0)

    overlay_img = (overlay_img*255).astype(np.uint8)
    return overlay_img
