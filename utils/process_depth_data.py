import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image 
from matplotlib import cm
import cv2
#load the depth data 
npy_file="/data/btinaz/mono_depth/data/session_2/0a0b55f7-f231-4578-b1c4-6353a0492ef5/depth.npy"
depth_data=np.load(npy_file)
print(np.min(depth_data),np.mean(depth_data), np.max(depth_data))



depthFrameColor = cv2.normalize(depth_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
depthFrameColor_c = cv2.equalizeHist(depthFrameColor)
depthFrameColor_c = cv2.applyColorMap(depthFrameColor_c, cv2.COLORMAP_INFERNO)
# cv2.imwrite('orig.jpg',depthFrameColor_t)
#depthFrameColor = cv2.medianBlur(depthFrameColor, 3)
depthFrameColor = cv2.equalizeHist(depthFrameColor)
kernel_close = np.ones((3,3),np.uint16)
kernel_open= np.ones((3,3),np.uint16)
closing = cv2.morphologyEx(depthFrameColor, cv2.MORPH_CLOSE,kernel_close)
closing_c = cv2.equalizeHist(closing)
closing_c = cv2.applyColorMap(closing_c, cv2.COLORMAP_INFERNO)
# cv2.imwrite('orig_open.jpg',depthFrameColor)
open_close=cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel_open)
open_close_c = cv2.equalizeHist(open_close)
open_close_c = cv2.applyColorMap(open_close_c, cv2.COLORMAP_INFERNO)
# cv2.imwrite('orig_open_close.jpg',depthFrameColor)

# open_close = cv2.applyColorMap(open_close, cv2.COLORMAP_INFERNO)
# opening=cv2.applyColorMap(opening, cv2.COLORMAP_INFERNO)
# depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_INFERNO)


# top=10
# bottom=10
# left=10
# right=10
# open_close = cv2.copyMakeBorder(open_close, top, bottom, left, right, cv2.BORDER_CONSTANT)
# opening = cv2.copyMakeBorder(opening, top, bottom, left, right, cv2.BORDER_CONSTANT)
# depthFrameColor = cv2.copyMakeBorder(depthFrameColor , top, bottom, left, right, cv2.BORDER_CONSTANT)



# closing=cv2.morphologyEx(depthFrameColor,cv2.MORPH_CLOSE,kernel_close)
# close_open=cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel_open)


#cv2.imwrite('Median_Filt_Close_Opening.jpg',opening)



concat_image=np.hstack((depthFrameColor_c,closing_c,open_close_c))
cv2.imwrite('Concat_image_type_kernel_5_close_open.jpg', concat_image)
#plt.imsave('Layout_Image.jpg',concat_image,cmap="jet",dpi=300)
#plt.imsave('Test.jpg',depth_data,cmap="jet",dpi=300)
#im = Image.fromarray(np.uint8(cm.gist_earth(depth_data)*255))

#print(depth_data.shape)



