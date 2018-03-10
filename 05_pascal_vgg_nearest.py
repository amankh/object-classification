from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt

from eval import compute_map
from tensorflow.core.framework import summary_pb2
import scipy.spatial.distance
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
#import models

tf.logging.set_verbosity(tf.logging.INFO)

plot_map = False
log_map = True


debug_data_loader = 0 # set to 1 to debug using prints
test_cnn = 0# set to 1 to train/test only 20 images
test_cnn_batch = 50
BATCH_SIZE = 10
NUM_ITERS = 50
LOG_DIR = 'logs/task4_1'

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
vgg_mapping = { 'vgg_16/conv1/conv1_1/biases': 'conv2d/bias',
            'vgg_16/conv1/conv1_1/weights': 'conv2d/kernel',
            'vgg_16/conv1/conv1_2/biases': 'conv2d_1/bias',
            'vgg_16/conv1/conv1_2/weights': 'conv2d_1/kernel',
            'vgg_16/conv2/conv2_1/biases': 'conv2d_2/bias',
            'vgg_16/conv2/conv2_1/weights': 'conv2d_2/kernel',
            'vgg_16/conv2/conv2_2/biases': 'conv2d_3/bias',
            'vgg_16/conv2/conv2_2/weights': 'conv2d_3/kernel',
            'vgg_16/conv3/conv3_1/biases': 'conv2d_4/bias',
            'vgg_16/conv3/conv3_1/weights': 'conv2d_4/kernel',
            'vgg_16/conv3/conv3_2/biases': 'conv2d_5/bias',
            'vgg_16/conv3/conv3_2/weights': 'conv2d_5/kernel',
            'vgg_16/conv3/conv3_3/biases': 'conv2d_6/bias',
            'vgg_16/conv3/conv3_3/weights': 'conv2d_6/kernel',
            'vgg_16/conv4/conv4_1/biases': 'conv2d_7/bias',
            'vgg_16/conv4/conv4_1/weights': 'conv2d_7/kernel',
            'vgg_16/conv4/conv4_2/biases': 'conv2d_8/bias',
            'vgg_16/conv4/conv4_2/weights': 'conv2d_8/kernel',
            'vgg_16/conv4/conv4_3/biases': 'conv2d_9/bias',
            'vgg_16/conv4/conv4_3/weights': 'conv2d_9/kernel',
            'vgg_16/conv5/conv5_1/biases': 'conv2d_10/bias',
            'vgg_16/conv5/conv5_1/weights': 'conv2d_10/kernel',
            'vgg_16/conv5/conv5_2/biases': 'conv2d_11/bias',
            'vgg_16/conv5/conv5_2/weights': 'conv2d_11/kernel',
            'vgg_16/conv5/conv5_3/biases': 'conv2d_12/bias',
            'vgg_16/conv5/conv5_3/weights': 'conv2d_12/kernel',
            'vgg_16/fc6/biases': 'conv2d_13/bias',
            'vgg_16/fc6/weights': 'conv2d_13/kernel',
            'vgg_16/fc7/biases': 'conv2d_14/bias',
            'vgg_16/fc7/weights': 'conv2d_14/kernel'}

class vgg_weights(tf.train.SessionRunHook):
     def begin(self):
         tf.contrib.framework.init_from_checkpoint('vgg_16.ckpt', vgg_mapping)

def summary_var(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = float(val)
    writer.add_summary(summary_proto, step)
    writer.flush()

def cnn_model_fn(features, labels, mode, num_classes=20):
    '''
    VGG config D ARCHITECTURE:
    -> image        [-1, 256, 256, 3]
    '''

    input_layer2 = tf.reshape(features["x"], [-1,256,256,3])
    weights = features["w"]
  
    #data augmentation
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_layer2 = tf.random_crop(input_layer2,
            size = [BATCH_SIZE,224,224,3])
        input_layer = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_layer2)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 224, 224), input_layer2)

    #conv layer1
    conv1 = tf.layers.conv2d(
        inputs= input_layer,
        filters = 64,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)

    #conv layer2
    conv2 = tf.layers.conv2d(
        inputs= conv1,
        filters = 64,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)

    #pool1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size =[2,2],
        strides=2)

    #conv layer3
    conv3 = tf.layers.conv2d(
        inputs= pool1,
        filters = 128,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)

    #conv layer4
    conv4 = tf.layers.conv2d(
        inputs=conv3 ,
        filters = 128,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)
    
    #pool2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv4,
        pool_size =[2,2],
        strides=2)

    #conv layer5
    conv5 = tf.layers.conv2d(
        inputs= pool2,
        filters = 256,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)
    
    #conv layer6
    conv6 = tf.layers.conv2d(
        inputs= conv5,
        filters = 256,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)

    #conv layer7
    conv7 = tf.layers.conv2d(
        inputs= conv6,
        filters = 256,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)

    #pool3
    pool3 = tf.layers.max_pooling2d(
        inputs=conv7,
        pool_size =[2,2],
        strides=2)

    #conv layer8
    conv8 = tf.layers.conv2d(
        inputs= pool3,
        filters = 512,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)
    
    #conv layer9
    conv9 = tf.layers.conv2d(
        inputs= conv8,
        filters = 512,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)

    #conv layer10
    conv10 = tf.layers.conv2d(
        inputs= conv9,
        filters = 512,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)
    
    #pool4
    pool4 = tf.layers.max_pooling2d(
        inputs=conv10,
        pool_size =[2,2],
        strides=2)

    #conv layer11
    conv11 = tf.layers.conv2d(
        inputs= pool4,
        filters = 512,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)
    
    #conv layer12
    conv12 = tf.layers.conv2d(
        inputs= conv11,
        filters = 512,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)

    #conv layer13
    conv13 = tf.layers.conv2d(
        inputs= conv12,
        filters = 512,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True)
    
    #pool5
    pool5 = tf.layers.max_pooling2d(
        inputs=conv13,
        pool_size =[2,2],
        strides=2)

    print(pool5)

    # fc layer 6
    dense1 = tf.layers.conv2d(
        inputs= pool5,
        filters = 4096,
        strides = 1,
        kernel_size = [7,7],
        padding= "valid",
        activation = tf.nn.relu,
        use_bias = True)
    
    
    #drropout1
    dropout1 = tf.layers.dropout(
        inputs = dense1,
        rate = 0.5,
        training = mode== tf.estimator.ModeKeys.TRAIN)
    
    # fc layer 7
    dense2 = tf.layers.conv2d(
        inputs=  dropout1,
        filters = 4096,
        strides = 1,
        kernel_size = [1,1],
        padding= "valid",
        activation = tf.nn.relu,
        use_bias = True)

    #drropout2
    dropout2= tf.layers.dropout(
        inputs = dense2,
        rate = 0.5,
        training = mode== tf.estimator.ModeKeys.TRAIN)

    dense2_flat = tf.reshape(dropout2,[-1,4096])
    
    #fc layer 3
    dense3 = tf.layers.dense(
        inputs= dense2_flat,
        units = 20)

    ## check this part
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=dense3, axis=1),
        "probabilities": tf.nn.sigmoid(dense3, name="sigmoid_tensor"),
        "fc7":dense2_flat,
        "pool5": tf.reshape(pool5, [-1,7*7*512]),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=dense3), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #learning schedule
        starter_learning_rate = 0.0001
        decay_steps = 1000

        learning_rate = tf.train.exponential_decay(
            learning_rate = starter_learning_rate,
            global_step = tf.train.get_global_step(),
            decay_steps = decay_steps,
            decay_rate = 0.5)
        tf.summary.scalar(name ="learning_rate", tensor=learning_rate)
        #tf.summary.image(name = "image", tensor =input_layer, max_outputs = BATCH_SIZE)
        

        optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum = 0.9)      
        grad_list = optimizer.compute_gradients(loss)
        for index, grad in grad_list:
            if index is not None:
                tf.summary.histogram("{}-grad".format(grad.name), index)

        #for index, grad in enumerate(grad_list):
        #    tf.summary.histogram("{}-grad".format(grad_list[index][1].name), grad_list[index])

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)



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

    return data, labels, weights 

def load_test(data_dir, split='test'):
    test_img = ['005458', '000045','000202','000006','000062','000001','000011','000116','000617','000014']
    number_of_images = 10
    data = np.empty((number_of_images, 256,256,3), dtype = "float32")
    labels = np.zeros((number_of_images, 20), dtype = "int")
    weights = np.zeros((number_of_images, 20), dtype = "int")


    for i in range(number_of_images):
        rel_path = test_img[i] + '.jpg'
        path_img =  data_dir +'/JPEGImages/' + rel_path 
        img = Image.open(path_img)
        img = img.resize((256,256), Image.ANTIALIAS)
        data[i,:,:,:] = np.asarray(img, dtype = "float32")
        img.close()
        del(img)

    return data, labels, weights



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
        return arr


def main():
    args = parse_args()

    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    test_data, test_labels, test_weights = load_test(
        args.data_dir, split='test')


    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
        num_classes=eval_labels.shape[1]),
        model_dir=LOG_DIR)
    
    #Logging
    tensors_to_log = {"loss": "loss", 
    "global_step" : "global_step"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data, "w": test_weights},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    for it in range(1):
        print('hello')
        print("SET :", it)
        eval_pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        test_pred = list(pascal_classifier.predict(input_fn=test_input_fn))
        eval_pool = np.stack([p['pool5'] for p in eval_pred])
        eval_fc = np.stack([p['fc7'] for p in eval_pred])
        test_pool = np.stack([p['pool5'] for p in test_pred])
        test_fc = np.stack([p['fc7'] for p in test_pred])

        nbrs_pool = NearestNeighbors(n_neighbors =3, algorithm ='ball_tree').fit(eval_pool)
        pool_dist, pool_ind = nbrs_pool.kneighbors(test_pool)

        nbrs_fc = NearestNeighbors(n_neighbors =3, algorithm ='ball_tree').fit(eval_fc)
        fc_dist, fc_ind = nbrs_fc.kneighbors(test_fc)

        print('*************')
        print('test data', test_data.shape)
        print('test pool', test_pool.shape)
        print('eval pool', eval_pool.shape)
        print('test fc ', test_fc.shape)
        print('eval fc', eval_fc.shape)
        print('fc_dist', fc_dist.shape)
        print('pool_dist', pool_dist.shape)

        print(fc_ind[:,1])

        fc_neigh1 = eval_data[fc_ind[:,1]]
        fc_neigh2 = eval_data[fc_ind[:,2]]
        pool_neigh1 = eval_data[pool_ind[:,1]]
        pool_neigh2 = eval_data[pool_ind[:,2]]


    np.save("logs/05_vgg_nn/05_vgg_test_images", test_data)
    np.save("logs/05_vgg_nn/05_vgg_pool_neigh1", pool_ind)
    np.save("logs/05_vgg_nn/05_vgg_fc_neigh1", fc_ind)






if __name__ == "__main__":
    main()