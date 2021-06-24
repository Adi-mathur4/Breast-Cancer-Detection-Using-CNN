# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:04:55 2020

@author: adity
"""

import numpy as np
import cv2

img=cv2.imread('D:/CNN/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-001.png',0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
c11 = clahe.apply(img)

cv2.imwrite('clahe_1.jpg',c11)
cv2.imshow('image',c11)
cv2.waitKey(0)
cv2.destroyAllWindows()