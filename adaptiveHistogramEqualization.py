#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# # Reading Images

# In[3]:


img_1 = cv2.imread('hw1_dark_road_1.jpg');
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2 = cv2.imread('hw1_dark_road_2.jpg');
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
img_3 = cv2.imread('hw1_dark_road_3.jpg');
img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)


# # Part a)

# # Function  for calculating histogram

# In[4]:


def histogram(img):
    size = img.shape
    numofpixels = size[0] * size[1];
    freq  = np.zeros((256,1));
    pdf   = np.zeros((256,1));
    for i in range(size[0]):

        for j in range(size[1]):

            value=img[i,j];

            freq[value]=freq[value]+1;

            pdf[value]=freq[value]/numofpixels;
    return freq


# # Calculating and Plotting Histogram

# In[5]:


freq_1 = histogram(img_1) 
plt.plot(freq_1)
plt.show
cv2.imshow('hw1_dark_road_1',img_1);
cv2.waitKey()


# In[6]:


freq_2 = histogram(img_2) 
plt.plot(freq_2)
plt.show
cv2.imshow('hw1_dark_road_2',img_2);
cv2.waitKey()


# In[7]:


freq_3 = histogram(img_3) 
plt.plot(freq_3)
plt.show
cv2.imshow('hw1_dark_road_3',img_3);
cv2.waitKey()


# # Part b)

# # Histogram Equalization Function

# In[8]:


def hist_equ(img):
    size = img.shape
    numofpixels = size[0] * size[1];
    freq  = np.zeros((256,1));
    pdf   = np.zeros((256,1));
# calculating freq and pdf    
    for i in range(size[0]):
        for j in range(size[1]):
            value=img[i,j];
            freq[value]=freq[value]+1;
            pdf[value]=freq[value]/numofpixels;
# Initializing arrays            
    hist_img = np.zeros((size[0],size[1]));
    hist_img = hist_img.astype(np.uint8)
    cdf   = np.zeros((256,1));
    summ  = np.zeros((256,1));
    output= np.zeros((256,1));
    sum=0;
    no_bins=255;
# calculating cdf    
    for i in range(np.size(pdf)):
        sum = sum+freq[i];
        summ[i]=sum;
        cdf[i]=summ[i]/numofpixels;
        output[i]=cdf[i]*no_bins;
    output = np.round_(output)
    output = output.astype(np.uint8)
# Reassigning values to input image
    for i in range(size[0]):
        for j in range(size[1]):
            hist_img[i,j]=output[img[i,j]];
    return hist_img


# # Histogram equalized images with thier histogram

# In[11]:


hist_equ_1 = hist_equ(img_1)
freq_1 = histogram(hist_equ_1) 
plt.plot(freq_1)
plt.show
cv2.imshow('Histogram equalization Image 1',hist_equ_1);
cv2.waitKey()


# In[12]:


hist_equ_2 = hist_equ(img_2)
freq_2 = histogram(hist_equ_2) 
plt.plot(freq_2)
plt.show
cv2.imshow('Histogram equalization Image 1',hist_equ_2);
cv2.waitKey()


# In[13]:


hist_equ_3 = hist_equ(img_3)
freq_3 = histogram(hist_equ_3) 
plt.plot(freq_3)
plt.show
cv2.imshow('Histogram equalization Image 1',hist_equ_3);
cv2.waitKey()


# # Part c)

# In[40]:


clahe_1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
local_hist_1 = clahe_1.apply(img_1)
cv2.imshow('Local Histogram equalizatized Image 1',local_hist_1);
cv2.waitKey()
freq_1 = histogram(local_hist_1) 
plt.plot(freq_1)
plt.show


# In[42]:


clahe_2 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
local_hist_2 = clahe_2.apply(img_2)
cv2.imshow('Local Histogram equalizatized Image 1',local_hist_2);
cv2.waitKey()
freq_2 = histogram(local_hist_2) 
plt.plot(freq_2)
plt.show


# In[45]:


clahe_3 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
local_hist_3 = clahe_3.apply(img_3)
cv2.imshow('Local Histogram equalizatized Image 1',local_hist_3);
cv2.waitKey()
freq_3 = histogram(local_hist_3) 
plt.plot(freq_3)
plt.show


# In[ ]:





# In[4]:





# In[ ]:





# In[ ]:





# In[ ]:




