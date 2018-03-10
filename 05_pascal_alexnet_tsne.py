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

#import models

tf.logging.set_verbosity(tf.logging.INFO)

plot_map = False
log_map = False


debug_data_loader = 0 # set to 1 to debug using prints
test_cnn = 0# set to 1 to train/test only 20 images
test_cnn_batch = 1000
BATCH_SIZE = 10
NUM_ITERS = 1
LOG_DIR = 'logs/task2_2'



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
    '''
    ALEXNET ARCHITECTURE:
    -> image        [-1, 256, 256, 3]
    -> data augmentation [-1,224,224,3]

    -> conv(11, 4, 96, 'VALID')  [-1, 256-11/4 +1 , ]
    -> relu()
    -> max_pool(3, 2)
    -> conv(5, 1, 256, 'SAME')
    -> relu()
    -> max_pool(3, 2)
    -> conv(3, 1, 384, 'SAME')
    -> relu()
    -> conv(3, 1, 384, 'SAME')
    -> relu()
    -> conv(3, 1, 256, 'SAME')
    -> max_pool(3, 2)
    -> flatten()
    -> fully_connected(4096)
    -> relu()
    -> dropout(0.5)
    -> fully_connected(4096)
    -> relu()
    -> dropout(0.5)
    -> fully_connected(20)
    '''

    input_layer2 = tf.reshape(features["x"], [-1,256,256,3])
    weights = features["w"]
  
    #data augmentation
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_layer = tf.random_crop(input_layer2,
            size = [BATCH_SIZE,224,224,3])
        input_layer = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_layer)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 224, 224), input_layer2)
        #input_layer = tf.image.crop_to_bounding_box(input_layer2,
        #    offset_height = 16, offset_width= 16,
        #    target_height = 224, target_width = 224)

    #conv layer1
    conv1 = tf.layers.conv2d(
        inputs= input_layer,
        filters = 96,
        strides = 4,
        kernel_size = [11,11],
        padding= "valid",
        activation = tf.nn.relu,
        use_bias = True,
        bias_initializer = tf.zeros_initializer(),
        kernel_initializer = tf.random_normal_initializer(0, 0.01))

    #pool1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size =[3,3],
        strides=2)

    #conv layer2
    conv2 = tf.layers.conv2d(
        inputs= pool1,
        filters = 256,
        strides = 1,
        kernel_size = [5,5],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True,
        bias_initializer = tf.zeros_initializer(),
        kernel_initializer = tf.random_normal_initializer(0, 0.01))

    #pool2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size =[3,3],
        strides=2)

    #conv layer3
    conv3 = tf.layers.conv2d(
        inputs= pool2,
        filters = 384,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True,
        bias_initializer = tf.zeros_initializer(),
        kernel_initializer = tf.random_normal_initializer(0, 0.01))

    #conv layer4
    conv4 = tf.layers.conv2d(
        inputs= conv3,
        filters = 384,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,
        use_bias = True,
        bias_initializer = tf.zeros_initializer(),
        kernel_initializer = tf.random_normal_initializer(0, 0.01))
    
    #conv layer5
    conv5 = tf.layers.conv2d(
        inputs= conv4,
        filters = 256,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu,  ## check if activation is there or not
        use_bias = True,
        bias_initializer = tf.zeros_initializer(),
        kernel_initializer = tf.random_normal_initializer(0, 0.01))

    #pool3
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size =[3,3],
        strides=2)

    #flatten
    pool3_flat = tf.reshape(pool3, [-1,5*5*256])

    # fc layer 1
    dense1 = tf.layers.dense(
        inputs = pool3_flat,
        units = 4096,
        activation = tf.nn.relu,
        use_bias = True,
        bias_initializer = tf.zeros_initializer(),
        kernel_initializer = tf.random_normal_initializer(0, 0.01))

    #drropout1
    dropout1 = tf.layers.dropout(
        inputs = dense1,
        rate = 0.5,
        training = mode== tf.estimator.ModeKeys.TRAIN)

    #fc layer 2
    dense2 = tf.layers.dense(
        inputs = dropout1,
        units = 4096,
        activation = tf.nn.relu,
        use_bias = True,
        bias_initializer = tf.zeros_initializer(),
        kernel_initializer = tf.random_normal_initializer(0, 0.01))
    
    #dropout2
    dropout2 = tf.layers.dropout(
        inputs = dense2,
        rate = 0.5,
        training = mode== tf.estimator.ModeKeys.TRAIN)

    #fc layer 2
    dense3 = tf.layers.dense(inputs = dropout2,units = 20,
        use_bias = True,
        bias_initializer = tf.zeros_initializer(),
        kernel_initializer = tf.random_normal_initializer(0, 0.01))


    ## check this part
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=dense3, axis=1),
        "probabilities": tf.nn.sigmoid(dense3, name="sigmoid_tensor"),
        "pool5" :pool3_flat,
        "fc7": dense2,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=dense3), name='loss')


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

        #learning schedule
        starter_learning_rate = 0.001
        decay_steps = 10000
        learning_rate = tf.train.exponential_decay(
            learning_rate = starter_learning_rate,
            global_step = tf.train.get_global_step(),
            decay_steps = decay_steps,
            decay_rate = 0.5)
        tf.summary.scalar(name ="learning_rate", tensor=learning_rate)
        #tf.summary.image(name = "image", tensor =input_layer, max_outputs = BATCH_SIZE)
        
        optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum = 0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    #eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

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

    print((eval_data.shape)[0])

    index_rand = np.random.randint(0,(eval_data.shape)[0], 1000)

    test_data = eval_data[index_rand]
    test_labels = eval_labels[index_rand]
    test_weights = eval_weights[index_rand]

    print(test_data.shape)


    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
        num_classes=train_labels.shape[1]),
        model_dir=LOG_DIR)
    
    #Logging
    tensors_to_log = {"loss": "loss", 
    "global_step" : "global_step"}
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

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data, "w": test_weights},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    test_pred = list(pascal_classifier.predict(input_fn=test_input_fn))
    test_fc = np.stack([p['fc7'] for p in test_pred])


    fc_tsne = TSNE(n_components =2).fit_transform(test_fc)
    print('fc_tsne', fc_tsne.shape)
    plt.figure(1)

    class_label = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    colormap = np.sum(test_labels*class_label,axis=1)/np.sum(test_labels, axis =1)

    print("colormap", colormap.shape)
    plt.scatter(fc_tsne[:,0], fc_tsne[:,1], c=colormap, cmap=plt.cm.get_cmap("jet", 20))
    plt.colorbar(ticks=range(20))
    plt.clim(0, 20)
    plt.show()




if __name__ == "__main__":
    main()