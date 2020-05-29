import os
import cv2
import random

from shutil import move as mv

dataset_dir	=	'/tmp/lfw/'
output_dir 	=	'/tmp/dataset/'
train_dir	=	os.path.join(output_dir,"train")
test_dir	=	os.path.join(output_dir,"test")

scale_factor= 4
img_size 	= 256

test_ratio = 0.2

def prepare_dataset(path):
	os.makedirs(os.path.join(path,train_dir,'hr'))
	os.makedirs(os.path.join(path,train_dir,'lr'))

	os.makedirs(os.path.join(path,test_dir,'hr'))
	os.makedirs(os.path.join(path,test_dir,'lr'))

	print('folder created successfuly')

def img_portioner(train_dir,ratio):
	file_num = len(os.listdir(os.path.join(train_dir,'hr')))
	test_num = int(file_num*ratio)

	rand_list = random.sample(range(0, file_num), test_num)

	images = os.listdir(os.path.join(os.path.join(train_dir,'hr')))

	for rand in rand_list:
		print('movin',images[rand],'to test folder')
		mv(os.path.join(train_dir,'hr',images[rand]),os.path.join(test_dir,'hr',images[rand]))
		mv(os.path.join(train_dir,'lr',images[rand]),os.path.join(test_dir,'lr',images[rand]))

print("program begin ...")

try:
	prepare_dataset("./")
except Exception as e:
	print(e)

name_list = os.listdir(dataset_dir)
for name in name_list:
	name_dir = os.path.join(dataset_dir,name)
	try:
		for img_file in os.listdir(name_dir):
			print('processing',img_file)
			img = cv2.imread(os.path.join(name_dir,img_file))

			img = cv2.resize(img,(img_size,img_size))
			cv2.imwrite(os.path.join(train_dir,'hr',img_file),img)

			img = cv2.resize(img,(img_size//scale_factor,img_size//scale_factor))
			cv2.imwrite(os.path.join(train_dir,'lr',img_file),img)
	except Exception as e:
		print(e)

try:
	img_portioner(train_dir,test_ratio)
except Exception as e:
	print(e)


print('dataset ready to use, Congratss cuy')