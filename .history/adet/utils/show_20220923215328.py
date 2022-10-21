import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os
from palettable.cartocolors.sequential import DarkMint_4, RedOr_3, BluYl_3, Emrld_2, Sunset_3
import pdb

def show_feature_map(feature_map, name):
    feature_map = feature_map.squeeze(1)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()

    for index in range(1, feature_map_num+1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap=DarkMint_4.mpl_colormap)
        plt.axis('off')
        # scipy.misc.imsave(str(index)+".png", feature_map[index-1])
    plt.savefig(os.path.join('feat_visualize', str(name)+".png"))
    plt.show()

def show_feature_map_v2(feature_map, name):
    feature_map = feature_map.squeeze(1)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.imshow(feature_map[index-1], cmap=DarkMint_4.mpl_colormap)
        plt.axis('off')
        plt.savefig(os.path.join('feat_visualize/'+str(name), str(index)+".png"))

def show_feature_map_v3(feature_map, name):
    feature_map = feature_map.squeeze(1)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.imshow(feature_map[index-1], cmap=RedOr_3.mpl_colormap)
        plt.axis('off')
        plt.savefig(os.path.join('feat_visualize/'+str(name), str(index)+".png"))

def show_feature_map_v4(feature_map, name):
    feature_map = feature_map.squeeze(1)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.imshow(feature_map[index-1], cmap=Sunset_3.mpl_colormap)
        plt.axis('off')
        plt.savefig(os.path.join('feat_visualize/'+str(name), str(index)+".png"))

def show_feature_map_heatmap(feature_map, name):
    feature_map = feature_map.squeeze(1)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.imshow(feature_map[index-1], cmap=plt.cm.jet)
        plt.axis('off')
        plt.savefig(os.path.join('feat_visualize/'+str(name)), 'heatmap' + str(index)+".png")
