import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt

def depth_vis(depth_raw):
    depthFrameColor = cv2.normalize(depth_raw, None, 255, 0, cv2.NORM_INF, cv2.CV_16UC1)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_INFERNO)
    return depthFrameColor


root_dir = '/data/btinaz/mono_depth/data/'
out_dir = '/data/btinaz/DatasetOpenCV/'

c = 1
thresh = 1e3
dim = (640,480)
bin_num = 100
kernel_close = np.ones((3,3),np.uint16)
kernel_open= np.ones((3,3),np.uint16)

for i in os.listdir(root_dir):
    print(i)
    for j in os.listdir(root_dir + '/' + i):
        if j == '.DS_Store':
            continue

        rgb_dir = root_dir + i + '/' + j + '/color.npy' 
        depth_dir = root_dir + i + '/' + j + '/depth.npy' 

        # load data
        try:
            rgb = np.load(rgb_dir)
            depth = np.load(depth_dir)
        except:
            continue

        # process color (resize to 640x480)
        rgb_resized = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)
        
        # process depth ()
        depth_resized = cv2.resize(depth, dim, interpolation = cv2.INTER_AREA)

        n, bins, _ =  plt.hist(depth_resized.flatten(), bin_num)
        cut_index = np.argwhere( n < thresh)[0]
        depth_resized[depth_resized > bins[cut_index]] = bins[cut_index]

        depth_closed = cv2.morphologyEx(depth_resized, cv2.MORPH_CLOSE, kernel_close)
        depth_final = cv2.morphologyEx(depth_closed, cv2.MORPH_OPEN, kernel_open)

        # save
        cv2.imwrite(out_dir + '/rgb_' + str(c) + '.jpg', rgb_resized)
        cv2.imwrite(out_dir + '/depth_' + str(c) + '.png', depth_final)

        # update index
        c += 1