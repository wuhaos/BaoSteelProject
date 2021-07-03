import numpy as np
import cv2 as cv

img = np.zeros((10,10)).astype(np.uint8)
img[5:7,6:8] = 7
img[2:4,2:4] = 1
print(img)
img = cv.resize(img,(3,8),interpolation=0)
print(img)
# img[(img==0) | (img == 1)] = 5
# print(img)