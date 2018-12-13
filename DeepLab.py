### Library function###############
import sys, os

import cv2
import numpy as np

sys.path.append('/home/developer/Desktop/sss/keras-deeplab-v3-plus-master/')

import model

from keras.applications.mobilenetv2 import preprocess_input



img = preprocess_input(cv2.imread('/home/developer/Desktop/DeepLab/group.jpg').astype("float"))
img = cv2.resize(img, (512,512))

import scipy.misc

image = scipy.misc.imread('/home/developer/Desktop/DeepLab/group.jpg')

img.max() # 0.9921875
model_dlv3 = model.Deeplabv3()

predicted = model_dlv3.predict(img[np.newaxis, ...])

person_score = predicted[0, :, : ,15]
back_score = predicted[0, :, :, 0]

mask = (person_score > back_score).astype("uint8") * 255
cv2.imwrite("/home/developer/Desktop/test.jpg", mask)

import matplotlib.pyplot as plt
image = scipy.misc.imread('/home/developer/Desktop/DeepLab/group.jpg')
def make_segmentation_mask(image, mask):
    imag = image.copy()
    imag[:,:,0] *= mask
    imag[:,:,1] *= mask
    imag[:,:,2] *= mask
    plt.imshow((imag * 255).astype(np.uint8))
    #plt.imshow(img)
    plt.imsave("/home/developer/Desktop/folder11.jpg",(imag * 255).astype(np.uint8))

make_segmentation_mask(image, mask)

