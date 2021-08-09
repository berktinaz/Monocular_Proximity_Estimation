import glob
import os.path
import random

root_dir = '/data/btinaz/DatasetOpenCV/'
foc = 518.8579
split = 0.7
train_dir = '/data/btinaz/mono_depth/custom_dataset_train.txt'
test_dir = '/data/btinaz/mono_depth/custom_dataset_test.txt'

files = glob.glob(os.path.join(root_dir, 'depth*'))
random.shuffle(files)

ind = round(len(files)*split)
train_set = files[:ind]
test_set = files[ind:]

with open(train_dir, 'w') as f:
    for fil in train_set:
        img_fil = fil.replace('depth','rgb') 
        img_fil = img_fil.replace('png','jpg')       
        f.write(img_fil + ' ' + fil + ' ' + str(foc) + '\n')

with open(test_dir, 'w') as f:
    for fil in test_set:
        img_fil = fil.replace('depth','rgb') 
        img_fil = img_fil.replace('png','jpg')       
        f.write(img_fil + ' ' + fil + ' ' + str(foc) + '\n')