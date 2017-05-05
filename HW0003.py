#HW0003
#2. print out the first image of train-images.idx3-ubyte package in szie 28x28 map
#3. print out the merged image from the first ten images of train-images.idx3-ubute in size 28x28 map
#4. caculate the mean of the first ten labels from train-lables.idx1-ubyte
#5. print out the first image from the train-iamges.idx3-ubyte in size 32x32

import os
import struct
import numpy as np
from PIL import Image

def ascii_show(image):
    for y in image:
        row = ""
        for  x in y:
            row += '{:02x}'.format(x)
        print (row)

def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError
    
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    
    return lbl, img

#print os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

lbl, img = read(dataset='training', path=".")
#ascii_show(img[0])

#3
pixels = np.zeros([28, 28])

for i in range(10):
    pixels += img[i]

pixels = pixels / 10
pixels = pixels.astype(np.uint8)
#ascii_show(pixels)

# 4. return the mean of the first ten labels
mean = 0
for i in range(10):
    mean += lbl[i]
mean = mean / 10
mean.astype(int)

#5. pad the first image with 0
padded_img0 = np.lib.pad(img[0], pad_width=(2, 2), mode='edge')
#ascii_show(padded_img0)



result = Image.fromarray(img[0])
result
result.save("img0.bmp")


























