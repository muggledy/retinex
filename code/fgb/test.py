from fast_gauss_blur import fastGaussBlur
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path

img=cv2.imread(os.path.join(os.path.dirname(__file__),'./img.png'))

plt.subplot(121)
plt.imshow(fastGaussBlur(img,10)[:,:,::-1])
plt.subplot(122)
plt.imshow(cv2.GaussianBlur(img,(0,0),10)[:,:,::-1])
plt.show()