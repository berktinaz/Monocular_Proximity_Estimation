import os

img_dir = '/data/NYUDEPTHV2/nyuv2-python-toolkit/NYUv2/image/test/'
depth_dir = '/data/NYUDEPTHV2/nyuv2-python-toolkit/NYUv2/depth/test/'
foc = 518.8579

out_dir = '/data/btinaz/mono_depth/AdaBins/train_test_inputs/nyudepthv2_custom_test_files_with_gt.txt'

with open(out_dir, 'w') as f:
    for ind in os.listdir(img_dir):
        f.write(img_dir + ind + ' ' + depth_dir + ind + ' ' + str(foc) + '\n')