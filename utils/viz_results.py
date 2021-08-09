import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pickle
import bbox_visualizer as bbv

def draw_rectangle(img,
                   bbox,
                   bbox_color=(0, 0, 255),
                   thickness=3,
                   is_opaque=False,
                   alpha=0.1):
    output = img.copy()
    if not is_opaque:
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      bbox_color, thickness)
    else:
        overlay = img.copy()

        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      bbox_color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output

def add_label(img,
              label,
              bbox,
              draw_bg=False,
              text_bg_color=(255, 255, 255),
              text_color=(255, 255, 255),
              scale=0.6,
              top=False):
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]

    if top:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] - 30]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]),
                          (label_bg[2] + 5, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + 5, bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 2)

    else:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] + 30]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]),
                          (label_bg[2] + 5, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + 5, bbox[1] - 5 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 2)

    return img

def process_single_image(num, typ):
    F = 518.8579
    imdir = '/data/btinaz/DatasetOpenCV/rgb_'+str(num)+'.jpg'
    if typ != 'baseline':
        depth_dr = '/data/btinaz/mono_depth/AdaBins/predictions_nyu_finetune2/__data__btinaz__DatasetOpenCV__rgb_'+str(num)+'.png'
    else:
        depth_dr = '/data/btinaz/mono_depth/AdaBins/predictions_nyu_baseline/__data__btinaz__DatasetOpenCV__rgb_'+str(num)+'.png'
    box_dr = '/data/digbose92/OpenCV_project/OpenCV_detection_pkl_files/rgb_'+str(num)+'.pkl'
    out_dr = './bbox_draw/'

    # read data
    color_img = cv2.imread(imdir)
    depth = cv2.imread(depth_dr, -1)
    with open(box_dr, 'rb') as f:
        box = pickle.load(f)

    # organize data and cast float values to integer
    boxes = box['bbox']
    labels = box['Class_labels']
    boxes = [i.astype('uint16') for i in boxes]

    # check if there are 2 boxes
    if len(boxes) != 2:
        return

    # draw boxes and add labels
    for i, box in enumerate(boxes):
        color_img = draw_rectangle(color_img, box)
        add_label(color_img, labels[i], box )

    # calculate box centers and draw on the image. Draw line between the centers
    centers = [0,0]
    for i, box in enumerate(boxes):
        temp = [int((box[0]+box[2])/2), int((box[1]+box[3])/2)]
        centers[i] = temp
        cv2.circle(color_img, temp, 1, (0,255,0), thickness=2, lineType=8, shift=0)

    cv2.line(color_img, centers[0], centers[1], (0,255,255), thickness=2)

    # get 3x3 depth values centered around centers
    depth_c = [0,0]
    for i, center in enumerate(centers):
        depth_c[i] = depth[center[1]-1:center[1]+2,center[0]-1:center[0]+2]/1000 #convert to meters
    print(depth_c)
    # get avg distance values
    depth_c_avg = [np.mean(i) for i in depth_c]

    # get x,y,z coordinates with the formula (u*Z/f, v*Z/f)
    cords = [0,0]
    for i, Z in enumerate(depth_c_avg):
        cords[i] = [centers[i][0]*Z/F, centers[i][1]*Z/F, Z]

    # calculate distance and draw over the line
    dist = np.sqrt((cords[0][0]-cords[1][0])**2+(cords[0][1]-cords[1][1])**2+(cords[0][2]-cords[1][2])**2)
    dist_str = '%s' % float('%.3g' % dist)
    print(dist_str)

    add_label(color_img, dist_str+'m', [int((centers[0][0]+centers[1][0])/2), int((centers[0][1]+centers[1][1])/2), 0, 0])

    # save image
    cv2.imwrite(out_dr + str(num) + typ + '.png', color_img)

num =  7556
typs = ['baseline', 'finetune']
for typ in typs:
    process_single_image(num, typ)