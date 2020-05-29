
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from Network import Generator, Discriminator
import Utils, Utils_model
from Utils_model import VGG_LOSS
# from test import one_image

from Utils import denormalize

from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
import numpy as np
import argparse, cv2

INPUT_DIR = "dataset/train/"
OUTPUT_DIR = "output/"
MODEL_SAVE_DIR = "model/" 

EPOCHS = 1000

IMAGE_SHAPE = (256,256,3)
BATCH_SIZE = 4
NUMBER_OF_IMAGES = 8000

TRAIN_TEST_RATIO = 0.8
RESUME_TRAINING = False
DOWNSCALE_FACTOR = 4

EPOCHS_CHECKPOINT = 1000

# ==============================================#

# resnet = arsitektur resnet
# inception = arsitektur inception
# inception-resnet = arsitektur inception-resnet
GENERATOR_ARCHITECTURE = "inception"

# ==============================================#

image_array = []

def gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    
    generator_result = generator(gan_input)
    gan_output = discriminator(generator_result)
    gan = Model(inputs= gan_input, outputs=[generator_result,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
    loss_weights=[1., 1e-3],
    optimizer = optimizer)

    return gan

def train(epochs, batch_size, input_dir, output_dir, model_save_dir, number_of_images, train_test_ratio, resume_train,downscale_factor,arch):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    
    hr_images,hr_label,lr_images,lr_label = Utils.load_training_data(input_dir,number_of_images)
    
    print(hr_images)
    loss = VGG_LOSS(IMAGE_SHAPE)
    lr_shape = (IMAGE_SHAPE[0]//downscale_factor, IMAGE_SHAPE[1]//downscale_factor,IMAGE_SHAPE[2])
    print(lr_shape)
    generator = Generator(lr_shape,downscale_factor,arch).generator()
    discriminator = Discriminator(IMAGE_SHAPE).discriminator()

    optimizer = Utils_model.get_optimizer()
    
    
    if(resume_train == True):
        last_epoch_number = Utils.get_last_epoch_number(model_save_dir+'last_model_epoch.txt')

        gen_model = model_save_dir+arch+"_gen_model"+str(last_epoch_number)+".h5"
        dis_model = model_save_dir+arch+"_dis_model"+str(last_epoch_number)+".h5"
        generator.load_weights(gen_model)
        discriminator.load_weights(dis_model)
        
    else:
        last_epoch_number = 1

    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = gan_network(discriminator, lr_shape, generator, optimizer, loss.vgg_loss)

    for e in range(last_epoch_number, last_epoch_number+epochs):
        print('-'*15, 'Epoch %d'%e,'-'*15)
        for _ in tqdm(range(1)):
            
            rand_nums = np.random.randint(0, hr_images.shape[0], size = batch_size)
            image_batch_hr = hr_images[rand_nums]
            image_batch_lr = lr_images[rand_nums]
            # video_images = lr_images[0]
            generated_images = generator.predict(image_batch_lr) #array of generated images

            real_data = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data = np.random.random_sample(batch_size)*0.2

            discriminator.trainable = True

            discriminator_loss_real = discriminator.train_on_batch(image_batch_hr, real_data)
            discriminator_loss_fake = discriminator.train_on_batch(generated_images, fake_data)
            discriminator_loss = 0.5 *np.add(discriminator_loss_fake,discriminator_loss_real) #Mean Of Discriminator Loss

            rand_nums = np.random.randint(0, hr_images.shape[0], size = batch_size)

            discriminator.trainable=False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,real_data])

        print("discriminator_loss : %f"%discriminator_loss)
        print("gan_loss : ",gan_loss)
        gan_loss = str(gan_loss)
        # generated_video_image = generator.predict(np.expand_dims(video_images,axis=0))
        Utils.save_losses_file(model_save_dir, e, gan_loss, discriminator_loss,arch+'_losses.txt')
        # image_array.append(cv2.cvtColor(denormalize(generated_video_image[0]),cv2.COLOR_BGR2RGB))
        
        if e % EPOCHS_CHECKPOINT == 0:
            Utils.save_losses_file(model_save_dir, e, gan_loss, discriminator_loss,'last_model_epoch.txt')
            generator.save(model_save_dir + arch+'_gen_model%d.h5'%e)
            discriminator.save(model_save_dir+arch+'_dis_model%d.h5'%e)
            # image_array.append(one_image)
            # Utils.save_video_file(str(e),image_array)
            
if __name__== "__main__":   
    train(int(EPOCHS/BATCH_SIZE), BATCH_SIZE, INPUT_DIR, OUTPUT_DIR, 
    MODEL_SAVE_DIR, NUMBER_OF_IMAGES, TRAIN_TEST_RATIO, RESUME_TRAINING, DOWNSCALE_FACTOR, GENERATOR_ARCHITECTURE)          