# Taken from https://www.tensorflow.org/tutorials/layers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.core.framework import summary_pb2


LOG_DIR = 'logs/task0'

def summary_var(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = float(val)
    writer.add_summary(summary_proto, step)
    writer.flush()


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
        activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.identity(tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        print('\n loss:', loss , '\n')
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=LOG_DIR)
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=200)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    test_eval_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)

    plt.figure(1)
    # plt.ion()

    total_steps = []
    eval_loss = []
    eval_acc = []
    
    for i in range(100):
        mnist_classifier.train(
            input_fn=train_input_fn,
            #steps=20000,
            steps = 300,
            hooks=[logging_hook])

        print ('\n')
        step2 = mnist_classifier.get_variable_value("global_step")

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)  
        #print('eval results:', eval_results, '\n')
        print('eval loss ', eval_results["loss"], 'step:',step2, '\n')
        print('********','\n')

        #writing map to tensorboard
        # summary_var(LOG_DIR, "Obtained_mAP", eval_results[""], it*NUM_ITERS)

        # total_steps = np.append(total_steps,step2)
        # eval_loss = np.append(eval_loss, eval_results["loss"])
        # eval_acc = np.append(eval_acc, eval_results["accuracy"])
        # plt.plot(total_steps, eval_loss, 'xr')
        # plt.plot(total_steps, eval_acc, 'xy')
        # plt.draw()
        # plt.pause(0.005)

    # plt.ioff()
    # plt.figure(2)
    # plt.plot(total_steps, eval_loss, '-r')
    # plt.savefig
    # plt.plot(total_steps, eval_acc, '-y')
    # plt.show()

    # lossFile = open('self_logs/eval_loss.txt','w')
    # accFile = open('self_logs/eval_acc.txt','w')
    # stepsFile = open('self_logs/steps_loss.txt','w')
    # for item in eval_loss:
    #     lossFile.write("%f \n" %item)
    # lossFile.close()
    # for item in eval_acc:
    #     accFile.write("%f \n" %item)
    # accFile.close()
    # for item in total_steps:
    #     stepsFile.write("%f \n" %item)
    # stepsFile.close()

if __name__ == "__main__":
    tf.app.run()
    