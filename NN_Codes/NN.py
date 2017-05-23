from sklearn.model_selection import KFold, cross_val_score
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import KFold, cross_val_score

import numpy as np
import tensorflow as tf
from parse_dataset import Crawler
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Parameters
    learning_rate = 0.001
    training_epochs = 50
    batch_size = 50
    display_step = 1

    # Network Parameters
    n_hidden_1 = 5 # 1st layer number of features
    #n_hidden_2 = 8 # 2nd layer number of features
    n_input = 28 # MNIST data input (img shape: 28*28)
    n_treatments = 3 # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_treatments])

    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        #layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        #out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        #'out': tf.Variable(tf.random_normal([n_hidden_2, n_treatments]))
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_treatments]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_treatments]))
    }

    objCrawler = Crawler("FinalCancer_Data.csv")
    data = objCrawler.parse_input()

    #find the maximum element
    max0 = np.amax(data[:, np.r_[0]])
    max1 = np.amax(data[:, np.r_[1]])
    max5 = np.amax(data[:, np.r_[5]])
    max6 = np.amax(data[:, np.r_[6]])
    max7 = np.amax(data[:, np.r_[7]])
    max8 = np.amax(data[:, np.r_[8]])
    max9 = np.amax(data[:, np.r_[9]])
    max10 = np.amax(data[:, np.r_[10]])
    max11 = np.amax(data[:, np.r_[11]])

    #Normalize Data
    for dataElements in data:
        dataElements[0] = dataElements[0]/max0
        dataElements[1] = dataElements[1]/max1
        dataElements[5] = dataElements[5]/max5
        dataElements[6] = dataElements[6]/max6
        dataElements[7] = dataElements[7]/max7
        dataElements[8] = dataElements[8]/max8
        dataElements[9] = dataElements[9]/max9
        dataElements[10] = dataElements[10]/max10
        dataElements[11] = dataElements[11]/max11


    trainInput = data[:, np.r_[0:28]]
    trainOutput = data[:, np.r_[28:31]]
    testInput = data[:, np.r_[0:28]]
    testOutput = data[:, np.r_[28:31]]

    '''print trainInput.shape
    print trainOutput.shape
    print testInput.shape
    print testOutput.shape'''

    train_size = len(trainInput)

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(train_size/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                #batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: trainInput,
                                                              y: trainOutput})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: testInput, y: testOutput}))
