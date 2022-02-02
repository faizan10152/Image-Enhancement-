#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# # Part a)

# In[6]:


orig_img_1 = cv2.imread('hw1_atrium.hdr') 
orig_img_2 = cv2.imread('hw1_memorial.hdr')
gray_img_1 = cv2.cvtColor(orig_img_1, cv2.COLOR_BGR2GRAY)
gray_img_2 = cv2.cvtColor(orig_img_2, cv2.COLOR_BGR2GRAY)
gray1_resize= cv2.resize(gray_img_1,(650,480))
gray2_resize= cv2.resize(gray_img_2,(650,480))
cv2.imshow('hw1_atrium',gray1_resize)
cv2.imshow('hw1_memorial',gray2_resize)
cv2.waitKey()


# # Part b)

# In[7]:


def gammacorrection(img,gamma):
    normalized = img/255
    exponential  = normalized ** gamma
    gm_corr = 255*exponential
    gm_corr = gm_corr.astype(np.uint8)
    return gm_corr


# In[26]:


gamma = 0.7
gm_cor_1 = gammacorrection(gray_img_1,gamma)
img = cv2.resize(gm_cor_1,(650,480))
cv2.imshow('frame1',gray1_resize)
cv2.imshow('gamma = 0.7',img)
cv2.waitKey()


# In[27]:


gm_cor_2 = gammacorrection(gray_img_2,gamma)
#img = cv2.resize(gray_img_2,(650,480))
cv2.imshow('frame1',gray_img_2)
cv2.imshow('gamma = 0.7',gm_cor_2)
cv2.waitKey()


# # Part c)

# In[36]:


gamma = 0.7
blue_1, green_1, red_1 = cv2.split(orig_img_1)
blue_cor_1 = gammacorrection(blue_1,gamma)
green_cor_1 = gammacorrection(green_1,0.3)
red_cor_1 = gammacorrection(red_1,gamma)
ch_cor_1 = np.dstack((blue_cor_1,green_cor_1,red_cor_1))
img = cv2.resize(ch_cor_1,(650,480))
cv2.imshow('frame1',gray1_resize)
cv2.imshow('green channel altered with gamma 0.3',img)
cv2.waitKey()


# In[39]:


blue_2, green_2, red_2 = cv2.split(orig_img_2)
blue_cor_2 = gammacorrection(blue_2,1)
green_cor_2 = gammacorrection(green_2,1)
red_cor_2 = gammacorrection(red_2,1.3)
ch_cor_2 = np.dstack((blue_cor_2,green_cor_2,red_cor_2))
cv2.imshow('original image',orig_img_2)
cv2.imshow('blue channel altered with gamma 1.5',ch_cor_2)
cv2.waitKey()


# In[85]:





# In[ ]:




