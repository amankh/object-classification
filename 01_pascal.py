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
#import models

LOG_DIR = 'logs/task1_1'

tf.logging.set_verbosity(tf.logging.INFO)

plot_map = False
log_map = True

debug_data_loader = 0 # set to 1 to debug using prints
test_cnn = 0# set to 1 to train/test only 20 images
test_cnn_batch = 200
BATCH_SIZE = 10
NUM_ITERS = 20

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

def summary_var(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = float(val)
    writer.add_summary(summary_proto, step)
    writer.flush()


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    input_layer = tf.reshape(features["x"], [-1,256,256,3])
    weights = features["w"]

    #conv layer1
    conv1 = tf.layers.conv2d(
        inputs= input_layer,
        filters = 32,
        kernel_size = [5,5],
        padding= "same",
        activation = tf.nn.relu)

    #pool1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size =[2,2],
        strides=2)

    #conv2
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu)

    #pooling layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = [2,2],
        strides = 2)

    #dense layers
    pool2_flat = tf.reshape(pool2, [-1, 64 * 64 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    # eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        


  
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss)



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
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
        num_classes=train_labels.shape[1]),
        model_dir=LOG_DIR)
    tensors_to_log = {"loss": "loss"}
    
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    if plot_map == True:
        plt.figure(1)
        plt.ion()
    if log_map == True:
        #gtmapFile = open('task1_map/gt_map.txt','w')
        #ranmapFile = open('task1_map/ran_map.txt','w')
        #evalmapFile = open('task1_map/eval_map.txt','w')
        meanmapFile = open('task1_mean_map.txt','w')


    for it in range(50):
        print("SET :", it)
        
        #train
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=NUM_ITERS,
            hooks=[logging_hook])
        
        #compute mAP
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        rand_AP = compute_map(
            eval_labels, np.random.random(eval_labels.shape),
            eval_weights, average=None)
        print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        #ground truth mAP
        gt_AP = compute_map(
            eval_labels, eval_labels, eval_weights, average=None)
        print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        for cid, cname in enumerate(CLASS_NAMES):
            print('{}: {}'.format(cname, _get_el(AP, cid)))

        #writing map to tensorboard
        summary_var(LOG_DIR, "Obtained_mAP", np.mean(AP), it*NUM_ITERS)
        
        if plot_map == True:
            plt.plot(it, np.mean(AP), 'xr')
            plt.draw()
            plt.pause(0.0001)

        if log_map  == True:
            meanmapFile.write("%f \n" %np.mean(AP))

    
    if plot_map == True: plt.ioff()
    if log_map == True: meanmapFile.close()





if __name__ == "__main__":
    main()
