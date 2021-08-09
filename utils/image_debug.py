import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import cv2


num = 10513
imdir = '/data/btinaz/DatasetOpenCV/rgb_'+str(num)+'.jpg'
depth_gt_dr = '/data/btinaz/DatasetOpenCV/depth_'+str(num)+'.png'
depth_ft_dr = '/data/btinaz/mono_depth/AdaBins/predictions_nyu_finetune2/__data__btinaz__DatasetOpenCV__rgb_'+str(num)+'.png'
depth_scr_dr = '/data/btinaz/mono_depth/AdaBins/predictions_nyu_finetune/__data__btinaz__DatasetOpenCV__rgb_'+str(num)+'.png'
depth_scr2_dr = '/data/btinaz/mono_depth/AdaBins/predictions_nyu_scratch_bs12_lr0.001/__data__btinaz__DatasetOpenCV__rgb_'+str(num)+'.png'
depth_bs_dr = '/data/btinaz/mono_depth/AdaBins/predictions_nyu_baseline/__data__btinaz__DatasetOpenCV__rgb_'+str(num)+'.png'

color_img = cv2.imread(imdir)
depth_gt = cv2.imread(depth_gt_dr, -1)
depth_ft = cv2.imread(depth_ft_dr, -1)
depth_scr = cv2.imread(depth_scr_dr, -1)
depth_scr2 = cv2.imread(depth_scr2_dr, -1)
depth_bs = cv2.imread(depth_bs_dr, -1)

depth_gt_nrm = cv2.normalize(depth_gt, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
depth_ft_nrm = cv2.normalize(depth_ft, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
depth_scr_nrm = cv2.normalize(depth_scr, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
depth_scr2_nrm = cv2.normalize(depth_scr2, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
depth_bs_nrm = cv2.normalize(depth_bs, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)

depth_gt_nrm = cv2.equalizeHist(depth_gt_nrm)
depth_ft_nrm = cv2.equalizeHist(depth_ft_nrm)
depth_scr_nrm = cv2.equalizeHist(depth_scr_nrm)
depth_scr2_nrm = cv2.equalizeHist(depth_scr2_nrm)
depth_bs_nrm = cv2.equalizeHist(depth_bs_nrm)

depth_gt_nrm = cv2.applyColorMap(depth_gt_nrm, cv2.COLORMAP_INFERNO)
depth_ft_nrm = cv2.applyColorMap(depth_ft_nrm, cv2.COLORMAP_INFERNO)
depth_scr_nrm = cv2.applyColorMap(depth_scr_nrm, cv2.COLORMAP_INFERNO)
depth_scr2_nrm = cv2.applyColorMap(depth_scr2_nrm, cv2.COLORMAP_INFERNO)
depth_bs_nrm = cv2.applyColorMap(depth_bs_nrm, cv2.COLORMAP_INFERNO)

cv2.imwrite('depth_gt.png', depth_gt_nrm)
cv2.imwrite('depth_ft.png', depth_ft_nrm)
cv2.imwrite('depth_scr.png', depth_scr_nrm)
cv2.imwrite('depth_scr2.png', depth_scr2_nrm)
cv2.imwrite('depth_bs.png', depth_bs_nrm)
cv2.imwrite('color.png', color_img)


# color_img = np.load(imdir)
# depth_raw = np.load(depthdir)
# color_img = np.asarray(Image.open(imdir))
# depth_2 = np.asarray(Image.open(depthimv2))

# depth_raw = cv2.resize(depth_raw, (640,480), interpolation = cv2.INTER_AREA)

# print(depth_2.shape, depth_2.max(), depth_2.min())
# print(depth_raw.shape, depth_raw.max(), depth_raw.min())
# n, bins, _ = plt.hist(depth_2.flatten(), 100)
# plt.savefig('temp1.png')
# print(n, bins)
# n, bins, _ =  plt.hist(depth_raw.flatten(), 100)
# print(n, bins)
# i = np.argwhere( n < 1e3)[0]
# print(i, bins[i])
# depth_raw[depth_raw > bins[i]] = bins[i]
# plt.figure()
# n, bins, _ =  plt.hist(depth_raw.flatten(), 100)
# print(n, bins)
# plt.savefig('temp2.png')

# depthFrameColor = cv2.normalize(depth_2, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
# depthFrameColor = cv2.equalizeHist(depthFrameColor)
# depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_INFERNO)

# cv2.imwrite('depth.png', depthFrameColor)
# cv2.imwrite('color.png', color_img)

