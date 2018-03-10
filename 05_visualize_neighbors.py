import numpy as np
import scipy.misc
import os.path as osp
import os
import matplotlib.pyplot as plt
import math
from PIL import Image
import cv2
import sys
import numpy as np
import argparse
from PIL import Image
from functools import partial


debug_data_loader = 0 # set to 1 to debug using prints
test_cnn = 0# set to 1 to train/test only 20 images
test_cnn_batch = 10

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]



def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that 
            are ambiguous.
    """
    # Wrote this function
    #loading data-list
    if split == "train":
        rel_path = "/ImageSets/Main/train.txt"
    elif split == "trainval":
        rel_path = "/ImageSets/Main/trainval.txt"
    elif split == "test":
        rel_path = "/ImageSets/Main/test.txt"
    data_list_file_path = data_dir + rel_path
    print('\n Location of', split,'data_list : ', data_list_file_path, '\n')
    data_list_file = open(data_list_file_path, 'r')
    data_list = data_list_file.readlines()
    data_list = [d.strip() for d in data_list if d.strip()]  #removing whitespaces and terminal characters
    data_list_file.close()

    #initializing data/labels/weight/ arrays
    number_of_images = len(data_list) #number of data points
    print(split ,'data size :', number_of_images, '\n')
    if test_cnn ==1:
        number_of_images = test_cnn_batch

    data = np.empty((number_of_images, 256,256,3), dtype = "float32")
    labels = np.empty((number_of_images, 20), dtype = "int")
    weights = np.empty((number_of_images, 20), dtype = "int")

    #loading images
    for i in range(number_of_images):
        
        rel_path = data_list[i] + '.jpg'
        path_img =  data_dir +'/JPEGImages/' + rel_path 
        if debug_data_loader == 1 :
            #path_img = data_dir + path_img
            print("data no :" , i, '\n')
            print('path to image :', path_img, '\n')

        
        img = Image.open(path_img)
        img = img.resize((256,256), Image.ANTIALIAS)
        
        data[i,:,:,:] = np.asarray(img, dtype = "float32")
        img.close()
        del(img)

    print("Loaded ", split, "images")


    # loading weights and labels
    for il in range(len(CLASS_NAMES)):
        rel_path = CLASS_NAMES[il]+'_' +split +'.txt' 
        path_class = data_dir + '/ImageSets/Main/' + rel_path
        #print ('class:', CLASS_NAMES[il], '| path:', path_class, '\n')

        temp_l = []
        temp_w = []
        with open(path_class, 'r') as class_file:
            for row in class_file:
                temp_x, temp_y = row.split()
                if int(temp_y) == -1:
                    temp_l = np.append(temp_l, int(0))
                    temp_w = np.append(temp_w, int(1))
                elif int(temp_y) == 1:
                    temp_l = np.append(temp_l, int(1))
                    temp_w = np.append(temp_w, int(1))
                elif int(temp_y) == 0:
                    temp_l = np.append(temp_l, int(1))
                    temp_w = np.append(temp_w, int(0))

        class_file.close()
        if len(temp_l) != number_of_images or len(temp_w) != number_of_images :
            print("************** \n WARNING \n size of labels/weights and data doesn't match",
            " \n **************")

        labels[:,il] = temp_l[0:number_of_images]
        weights[:, il] = temp_w[0:number_of_images]
        del( temp_x,temp_y, temp_l, temp_w)
        if debug_data_loader == 1:
            print(labels[1:20,:])
            print(labels[number_of_images-20:number_of_images+100,:])
            print(labels[number_of_images-1,:])

    if debug_data_loader == 1:
        #print(np.size(labels))
        print(np.shape(weights))
        print(np.shape(labels))
        print(labels[0:10,:])
        print(weights[0:10,:])

    print("Loaded ", split, "weights and labels")

    return data, labels, weights,data_list 







def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arrs    



def main():

	print('hey')
	args = parse_args()
	data_dir = args.data_dir 
	eval_data, eval_labels, eval_weights, eval_data_list = load_pascal(args.data_dir, split='test')
	pool_neigh1=np.load("logs/05_vgg_nn/05_vgg_pool_neigh1.npy")
	fc_neigh1=np.load("logs/05_vgg_nn/05_vgg_fc_neigh1.npy")
	

	test_img = ['005458', '000045','000202','000006','000062','000001','000011','000116','000617','000014']
	print(pool_neigh1.shape)

	for it in range(10):
		print(it)
		print(test_img[it])
		i = str(it)
		print(i)
		out_path_test = ('logs/05_nn/' +i+'test_img.jpg')
		print(out_path_test)
		img = Image.open(data_dir + '/JPEGImages/' + test_img[it] +'.jpg')
		img.save(out_path_test)
		img.close()

		# print(pool_neigh1[it,1])
		ind_pool1  = eval_data_list[pool_neigh1[it,1]]
		ind_pool2  = eval_data_list[pool_neigh1[it,2]]
		ind_pool0  = eval_data_list[pool_neigh1[it,0]]

		ind_fc1  = eval_data_list[fc_neigh1[it,1]]
		ind_fc2  = eval_data_list[fc_neigh1[it,2]]
		ind_fc0  = eval_data_list[fc_neigh1[it,0]]

		img = Image.open(data_dir + '/JPEGImages/' + ind_pool0 +'.jpg')
		out_path_test = ('logs/05_nn/' +i+'pool0_nn.jpg')
		img.save(out_path_test)
		img.close()

		img = Image.open(data_dir + '/JPEGImages/' + ind_pool1 +'.jpg')
		out_path_test = ('logs/05_nn/'+i+'pool1_nn.jpg')
		img.save(out_path_test)
		img.close()
		img = Image.open(data_dir + '/JPEGImages/' + ind_pool2 +'.jpg')
		out_path_test = ('logs/05_nn/'+i+'pool2_nn.jpg')
		img.save(out_path_test)
		img.close()
		# print(ind_pool2)
		# print(ind_pool0)

		img = Image.open(data_dir + '/JPEGImages/' + ind_fc0 +'.jpg')
		out_path_test = ('logs/05_nn/'+i+'fc0_nn.jpg')
		img.save(out_path_test)
		img.close()

		img = Image.open(data_dir + '/JPEGImages/' + ind_fc1 +'.jpg')
		out_path_test = ('logs/05_nn/'+i+'fc1_nn.jpg')
		img.save(out_path_test)
		img.close()
		img = Image.open(data_dir + '/JPEGImages/' + ind_fc2 +'.jpg')
		out_path_test = ('logs/05_nn/'+i+'fc2_nn.jpg')
		img.save(out_path_test)
		img.close()


		# print(eval_data_list[pool_neigh1[it,1]])




	
        

    


    




if __name__ == "__main__":
	main()