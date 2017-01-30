import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

"""
Purpose: Use logistic regression to classify mnist dataset
Optimizer: GradientDescentOptimizer
Learning Rate: 0.003
Achieved test accuracy: 92.6%
"""

# Get the data
mnist = read_data_sets("data", one_hot=True, validation_size=0, reshape=True)

# Define placeholders for the training and test data
X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# Define variables needed for weights and biases
W = tf.Variable(tf.zeros([28*28, 10]))
b = tf.Variable(tf.zeros([10]))

# Define logit
y_hat = tf.add(tf.matmul(X, W), b)

# Define the cost function
cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(tf.sigmoid(y_hat), axis=1),
                              tf.argmax(tf.cast(y_true, dtype=tf.float32), axis=1))
acc = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# Define optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(cost)

# Define operator to initialize variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(10000):
        X_batch, y_batch = mnist.train.next_batch(100)
        sess.run(train_step,
                 feed_dict={
                     X: X_batch,
                     y_true: y_batch
                 })
    test_acc = sess.run(acc,
                        feed_dict={
                            X: mnist.test.images,
                            y_true: mnist.test.labels
                        })
    print("test accuracy = {0}".format(test_acc))