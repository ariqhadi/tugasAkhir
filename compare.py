from math import log10, sqrt 
import cv2 
import numpy as np 
from skimage.measure import compare_ssim

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def SSIM(original, compressed): 
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    return score

def main(): 
     original = cv2.imread("original_image.png") 
     compressed = cv2.imread("compressed_image.png", 1) 

     psnr = PSNR(original, compressed) 
     ssim = SSIM(original, compressed)
       
if __name__ == "__main__": 
    main() 