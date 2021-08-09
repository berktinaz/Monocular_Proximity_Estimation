import matplotlib.pyplot as plt
import cv2
import pickle
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import model_io
import utils
from models import UnetAdaptiveBins

from time import time

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class InferenceHelper:
    def __init__(self, dataset='nyu', device='cuda:0'):
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = 5e-1
            self.max_depth = 6
            self.saving_factor = 1000  # used to save in 16 bit
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "/data/btinaz/mono_depth/AdaBins/checkpoints/FineTuneTest2_27-Jul_23-49-nodebs8-tep5-lr0.00017-wd0.1-c05e3444-661b-4061-87dc-1ac128ec057e_best.pt" #finetune
        elif dataset == 'kitti':
            self.min_depth = 1e-3
            self.max_depth = 80
            self.saving_factor = 256
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_kitti.pt"
        else:
            raise ValueError("dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))

        model, _, _ = model_io.load_checkpoint(pretrained_path, model)
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        img = np.asarray(pil_image) / 255.

        img = self.toTensor(img).unsqueeze(0).float().to(self.device)
        bin_centers, pred = self.predict(img)

        if visualized:
            viz = utils.colorize(torch.from_numpy(pred).unsqueeze(0), vmin=None, vmax=None, cmap='inferno')
            pred = np.asarray(pred*1000, dtype='uint16')
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred

    @torch.no_grad()
    def predict(self, image):
        bins, pred = self.model(image)
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # Flip
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred_lr = self.model(image)[-1]
        pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
                                          mode='bilinear', align_corners=True).cpu().numpy()

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]

        return centers, final

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        transform = ToTensor()
        all_files = glob.glob(os.path.join(test_dir, "*"))
        self.model.eval()
        for f in tqdm(all_files):
            image = np.asarray(Image.open(f), dtype='float32') / 255.
            image = transform(image).unsqueeze(0).to(self.device)

            centers, final = self.predict(image)
            # final = final.squeeze().cpu().numpy()

            final = (final * self.saving_factor).astype('uint16')
            basename = os.path.basename(f).split('.')[0]
            save_path = os.path.join(out_dir, basename + ".png")

            Image.fromarray(final).save(save_path)

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
              text_color=(255, 0, 0),
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

def process_single_image(color_img, depth, box):
    F = 518.8579

    # organize data and cast float values to integer
    boxes = box['bbox']
    labels = box['Class_labels']
    boxes = [i.astype('uint16') for i in boxes]

    # Continue if there is at most 2 boxes
    if len(boxes) > 2:
        return color_img, -1
    
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
    
    # check if there are 2 boxes
    if len(boxes) != 2:
        return color_img, -1

    cv2.line(color_img, centers[0], centers[1], (0,255,255), thickness=2)

    # get 3x3 (or 5x5) depth values centered around centers
    depth_c = [0,0]
    for i, center in enumerate(centers):
        # depth_c[i] = depth[center[1]-1:center[1]+2,center[0]-1:center[0]+2]
        depth_c[i] = np.median(depth[center[1]-2:center[1]+3,center[0]-2:center[0]+3])

    # get avg distance values
    depth_c_avg = [np.mean(i) for i in depth_c]

    # get x,y,z coordinates with the formula (u*Z/f, v*Z/f)
    cords = [0,0]
    for i, Z in enumerate(depth_c_avg):
        cords[i] = [centers[i][0]*Z/F, centers[i][1]*Z/F, Z]

    # calculate distance and draw over the line
    dist = np.sqrt((cords[0][0]-cords[1][0])**2+(cords[0][1]-cords[1][1])**2+(cords[0][2]-cords[1][2])**2)
    dist_str = '%s' % float('%.3g' % dist)
    # print(dist_str)

    add_label(color_img, dist_str+'m', [int((centers[0][0]+centers[1][0])/2), int((centers[0][1]+centers[1][1])/2), 0, 0])

    # save image
    return color_img, dist

##### MAIN ######

video_dir = '/data/btinaz/mono_depth/final_videos/session_2.mp4'
box_dr = '/data/digbose92/OpenCV_project/final_videos_pkl_files/session_2.pkl'
with open(box_dr, 'rb') as f:
    boxes = pickle.load(f)

dim = (640,480)

cap = cv2.VideoCapture(video_dir)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(video_dir[:-4] + '_processed.mp4',fourcc, 30, (640*2,480))

inferHelper = InferenceHelper()

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Initialize index and distance array 
idx = 0
dists = []

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, img = cap.read()
  if ret == True:

    # Resize the image
    img_r = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # Pass the image through the network
    start = time()
    centers, pred = inferHelper.predict_pil(img_r)
    # print(time()-start)
    depth = pred.squeeze().squeeze()

    # Get the bounding box for this frame
    box = boxes["frame_" + str(idx)]
    idx += 1

    # Process single image and append distance to the array
    img_proc, dist = process_single_image(img_r.copy(), depth, box)
    dists.append(dist)

    # Process depth image
    depth_n = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depth_ne = cv2.equalizeHist(depth_n)
    depth_nec = cv2.applyColorMap(depth_ne, cv2.COLORMAP_INFERNO)
    
    # Stack processed image and depth image side-by-side
    # print(img_proc.shape, depth_nec.shape)
    final_f = np.hstack((img_proc, depth_nec))

    # Write the cascaded image
    out.write(final_f)

    # cv2.imwrite('test.png', final_f)
  # Break the loop
  else: 
    break

# When finished release video capture and writer objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()

np.save('/data/btinaz/mono_depth/final_videos/distances_2_5x5_baseline.npy', dists)
