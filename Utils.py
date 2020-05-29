

from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy.misc import imresize
import os
import sys
import subprocess
import re
from Network import Generator
from tqdm import trange

import cv2
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def save_video_file(title,img_array):
    out = cv2.VideoWriter(title+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (256,256))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
        
def generate_two_plot(sr_image,lr_image,output_dir):
    
    dim=(1,2)
    figsize=(10,5)
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(sr_image[0], interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(lr_image, interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir)

def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
   
def load_training_data(directory,num_of_img,reverse = False):
    
    hr_filename = os.listdir(os.path.join(directory,'hr'))
    lr_filename = os.listdir(os.path.join(directory,'lr'))

    hr_images = []
    hr_label= []

    lr_images = []
    lr_label= []
    
    for i in trange(num_of_img,desc='Loading HR images'):
        if reverse:
            i = num_of_img - i    
        img = data.imread(os.path.join(directory,'hr',hr_filename[i]))
        img = normalize(img)

        hr_images.append(img)
        hr_label.append(1 - np.random.uniform(low=0, high=0.2))

    for i in trange(num_of_img,desc='Loading LR images'):
        if reverse:
            i = num_of_img - i
        img = data.imread(os.path.join(directory,'lr',lr_filename[i]))
        img = normalize(img)

        lr_images.append(img)
        lr_label.append(np.random.uniform(low=0, high=0.2))

    return array(hr_images),hr_label,array(lr_images),lr_label

def get_last_epoch_number(log_of_losses):
    last_train = subprocess.check_output(['tail', '-1', log_of_losses])
    epoch_number = re.search(r'\d+', str(last_train)).group()

    return int(epoch_number)
    
def save_losses_file(save_dir, e, gan_loss, discriminator_loss,filename):
    loss_file = open(save_dir+filename,'a+')
    loss_file.write('epoch %d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss))
    loss_file.close()