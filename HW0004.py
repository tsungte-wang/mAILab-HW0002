#HW0004

import os
import struct
import numpy as np
import matplotlib.pyplot as plt

# the following codes are derived from https://gist.github.com/akesling/5358964
def ascii_show(image):
    for y in image:
        row = ""
        for x in y:
            row += '{:02x}'.format(x)
        print(row)

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

lbl, img = read(dataset="training", path=".")

def make_convoluted_images(imgs, num_imgs, pad_width, sobel_filter):
    convoluted_images_list = []
    
    for img_idx in range(num_imgs):
        image = imgs[img_idx]
        padded_image = np.lib.pad(image, pad_width, mode='edge')
        convoluted_image = np.zeros([28,28])
        
        size_filter = sobel_filter.shape[0]
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                convoluted_image[i, j] = np.sum(np.multiply(padded_image[i:i+size_filter, j:j+size_filter], sobel_filter), axis=None)
        
        convoluted_image = convoluted_image
        convoluted_images_list.append(convoluted_image)
    
    return convoluted_images_list

def plot_images(image_list):
    L = len(image_list)
    
    for i in range(L):
        plt.subplot(1, L, i + 1)
        plt.imshow(image_list[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])


convoluted_images_Gx = make_convoluted_images(img, 5, (1, 1), Gx)
convoluted_images_Gy = make_convoluted_images(img, 5, (1, 1), Gy)

#plot_images(convoluted_images_Gx)

#plot_images(convoluted_images_Gy)

def make_sobel_filter(n):
    filter_ = np.zeros([n, n])
    
    first_col = np.array([range(n // 2, n), range(n-2, n // 2 - 1, -1)])
    first_col = np.concatenate(first_col, axis=0)
    
    for column in range(n//2):
        filter_[0:n, column] = first_col
        filter_[0:n, n - column -1] = -first_col
        first_col -= 1
    
    return filter_

img32x32_list = make_convoluted_images(img, 1, (2, 2), make_sobel_filter(5))
img32x32 = img32x32_list[0].astype("uint8")
#ascii_show(img32x32)

#plot_images(img32x32_list)


img34x34_list = make_convoluted_images(img, 1, (3, 3), make_sobel_filter(7))
img34x34 = img34x34_list[0].astype("uint8")
#ascii_show(img34x34)

#plot_images(img34x34_list)

img36x36_list = make_convoluted_images(img, 1, (4, 4), make_sobel_filter(9))
img36x36 = img36x36_list[0].astype("uint8")
#ascii_show(img36x36)

plot_images(img36x36_list)








