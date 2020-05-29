from keras.models import load_model

import cv2
import numpy as np
import Utils as Utils
from Utils_model import VGG_LOSS, get_optimizer
from Network import Generator

INPUT_DIR = "dataset/test/lr/Dan_Prinster_0001.jpg"
OUTPUT_DIR = "output/output.jpg"

MODEL_DIR = "model/"
DOWNSCALE_FACTOR = 4

# kalau pake model factor 8 kali DOWNSCALE_FACTOR nya juga ganti


def one_image():
    lr_shape = (64,64,3)
    loss = VGG_LOSS(lr_shape)
    optimizer = get_optimizer()

    last_epoch_number = Utils.get_last_epoch_number(MODEL_DIR+'last_model_epoch.txt')
    gen_model = MODEL_DIR+"inception_gen_model"+str(last_epoch_number)+".h5"

    generator = Generator(lr_shape,DOWNSCALE_FACTOR,"inception").generator()
    generator.load_weights(gen_model)
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    
    Utils.generate_one_image(INPUT_DIR,generator,OUTPUT_DIR)

def plot_image():
    lr_shape = (64,64,3)
    loss = VGG_LOSS(lr_shape)
    optimizer = get_optimizer()
    
    image = cv2.imread(INPUT_DIR)
    image_lr = Utils.normalize(image)
    
    last_epoch_number = Utils.get_last_epoch_number(MODEL_DIR+'last_model_epoch.txt')
    gen_model = MODEL_DIR+"inception_gen_model"+str(last_epoch_number)+".h5"

    generator = Generator(lr_shape,DOWNSCALE_FACTOR,"inception").generator()
    generator.load_weights(gen_model)
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)

    gen_img = generator.predict(np.expand_dims(image_lr,axis=0))
    sr_image = Utils.denormalize(gen_img)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Utils.generate_two_plot(sr_image,image,OUTPUT_DIR)

one_image()




