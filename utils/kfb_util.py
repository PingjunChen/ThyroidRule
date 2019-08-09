import os, sys, pdb
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color, filters
from skimage import img_as_ubyte
from skimage import io
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from shapely.geometry import Polygon
import cv2
import math

FileAbsPath = os.path.abspath(__file__)
ProjectPath = os.path.dirname(os.path.dirname(FileAbsPath))
sys.path.append(os.path.join(ProjectPath, 'kfb'))
import kfb_deepzoom, kfbslide


def load_kfb2arr(kfb_filepath, level=0, wsi_dim=None):
    kfb_slide = kfb_deepzoom.KfbDeepZoomGenerator(kfbslide.KfbSlide(kfb_filepath))
    tile_index = kfb_slide._dz_levels - 1 - level
    x_count, y_count = kfb_slide._t_dimensions[tile_index]
    # substract 1 to crop boundary
    x_count, y_count = x_count - 1, y_count - 1
    x_dim, y_dim = kfb_slide._z_dimensions[tile_index]

    assert x_count*256 <= x_dim and y_count*256 <= y_dim

    wsi_img = np.zeros((y_count*256, x_count*256, 3)) # Crop boundary
    for index_x in range(x_count):
        for index_y in range(y_count):
            start_x, start_y = index_x*256, index_y*256
            wsi_img[start_y:start_y+256, start_x:start_x+256, :] = kfb_slide.get_tile(tile_index, (index_x, index_y))
    wsi_img = wsi_img / 255.0

    # Select regions
    if wsi_dim != None:
        wsi_img = wsi_img[:wsi_dim[0], :wsi_dim[1], :]

    # plt.imshow(wsi_img)
    # plt.show()
    return wsi_img


# Locate tissue regions from kfb low level image
def find_tissue_cnt(kfb_path, level=4, thresh_val=0.82, min_size=2.0e5):
    wsi_img = load_kfb2arr(kfb_path, level)
    # Gray
    gray = color.rgb2gray(wsi_img)
    # Smooth
    smooth = filters.gaussian(gray, sigma=9)
    # Threshold
    binary = smooth < thresh_val
    # Fill holes
    fill = binary_fill_holes(binary)
    # Remove outliers
    mask = remove_small_objects(fill, min_size=min_size, connectivity=8)
    # Find contours
    _, cnts, _ = cv2.findContours(img_as_ubyte(mask),
                                  mode=cv2.RETR_EXTERNAL,
                                  method=cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(wsi_img, cnts, -1, (255, 0, 0), 8)
    # plt.imshow(wsi_img)
    # plt.show()

    return wsi_img, mask, cnts


def split_regions(kfb_path, img_level=3, cnt_level=4):
    s_img, mask, cnts = find_tissue_cnt(kfb_path, cnt_level)
    img_cnt_ratio = 2**(cnt_level-img_level)
    wsi_dim = [ele*img_cnt_ratio for ele in s_img.shape[:2]]
    wsi_img = load_kfb2arr(kfb_path, img_level, wsi_dim)

    RAW_SIZE = 299
    SIZE1, SIZE2, SIZE4 = int(RAW_SIZE/4), int(RAW_SIZE/2), RAW_SIZE
    split_arr, patch_list = [], []
    for c_ind in range(len(cnts)):
        cur_cnt = cnts[c_ind] * img_cnt_ratio
        cur_cnt = np.squeeze(cur_cnt)
        w_coors = [int(round(ele)) for ele in cur_cnt[:, 0].tolist()]
        h_coors = [int(round(ele)) for ele in cur_cnt[:, 1].tolist()]
        w_min, w_max = min(w_coors), max(w_coors)
        h_min, h_max = min(h_coors), max(h_coors)

        # Width range to crop
        start_w = (w_min - SIZE1) if (w_min - SIZE1) > 0 else 0
        num_w = int(math.ceil((w_max - start_w - SIZE2)/SIZE2))
        # Height range to crop
        start_h = (h_min - SIZE1) if (h_min - SIZE1) > 0 else 0
        num_h = int(math.ceil((h_max - start_h - SIZE2)/SIZE2))

        poly_cnt = Polygon(cur_cnt)
        for iw in range(0, num_w):
            for ih in range(0, num_h):
                # determine current rectangular is inside the contour or not
                cur_coors = [(start_w+iw*SIZE2, start_h+ih*SIZE2), (start_w+iw*SIZE2+SIZE4, start_h+ih*SIZE2),
                             (start_w+iw*SIZE2+SIZE4, start_h+ih*SIZE2+SIZE4), (start_w+iw*SIZE2, start_h+ih*SIZE2+SIZE4)]
                try:
                    if poly_cnt.contains(Polygon(cur_coors)):
                        split_arr.append((start_h+ih*SIZE2, start_w+iw*SIZE2))
                        split_patch = wsi_img[start_h+ih*SIZE2:start_h+ih*SIZE2+SIZE4, start_w+iw*SIZE2:start_w+iw*SIZE2+SIZE4, :]
                        patch_list.append(split_patch)
                except:
                    print("Error in Polygon relationship")
    return split_arr, patch_list, wsi_dim, s_img, mask
