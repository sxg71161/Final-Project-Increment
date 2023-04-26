import cv2
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr

def computeSimilarity(imageA, imageB):
    return ssim(cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY),cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY))

img = cv2.imread('before.jpeg')
median = cv2.medianBlur(img, 5)
compare = np.concatenate((img, median), axis=1) #side by side comparison

psnr = PSNR(img, median)
ssim = computeSimilarity(img, median)
print(str(psnr)+" "+str(ssim))


cv2.imshow('img', compare)
