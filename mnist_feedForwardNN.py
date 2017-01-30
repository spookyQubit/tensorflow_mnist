import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math

"""
Purpose: Use neural nets to classify mnist dataset
Network: 3 layers
         layer1: input=28*28, output=500
         layer2: input=500, output=100
         layer3: input=100, output=10
Decrease learning rate while training: Reduces variance
Achieved test accuracy: 96.4%

Yet to Implement:
1) Use dropout: Reduce over fitting
2) Batch normalization ??
"""

# Get data
mnist = read_data_sets("data",
                       one_hot=True,
                       validation_size=0,
                       reshape=True)

# Define placeholders for inputs and outputs
X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# Define variables and logit
# Layer 1
W1 = tf.Variable(tf.truncated_normal(shape=[28*28, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]))
z1 = tf.matmul(X, W1) + b1
a1 = tf.nn.relu(z1)
# Layer 2
W2 = tf.Variable(tf.truncated_normal(shape=[500, 100], stddev=0.1))
b2 = tf.Variable(tf.zeros([100]))
z2 = tf.matmul(a1, W2) + b2
a2 = tf.nn.relu(z2)
# Layer 3
W3 = tf.Variable(tf.truncated_normal(shape=[100, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
y_hat = tf.matmul(a2, W3) + b3

# Define cost
cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true)

# Define optimizer
lr = tf.placeholder(tf.float32)
max_lr = 0.003
min_lr = 0.001
decay_speed = 2000
train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(tf.cast(y_true, tf.float32), axis=1),
                              tf.argmax(tf.sigmoid(y_hat), axis=1))
acc = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# Define operator to initialize variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(1000):
        learning_rate = min_lr + (max_lr - min_lr) * math.exp(-i/decay_speed)
        X_batch, y_batch = mnist.train.next_batch(100)
        sess.run(train_step,
                 feed_dict={
                     X: X_batch,
                     y_true: y_batch,
                     lr: learning_rate
                 })

    test_accuracy = sess.run(acc,
                             feed_dict={
                                 X: mnist.test.images,
                                 y_true: mnist.test.labels
                             })
    print("test accuracy = {0}".format(test_accuracy))
