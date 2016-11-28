import tensorflow as tf

# MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)

# Learning Setting
iterations = 2000
batch_size = 100
dropout_rate = 0.5

# Graph input
x = tf.placeholder(tf.float32, [None, 784])  # image shape: 28*28
y = tf.placeholder(tf.float32, [None, 10])  # number of class: 10
keep_prob = tf.placeholder(tf.float32)  # dropout


# Layer wrapper
def my_conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def my_maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# Weight Creation
def weights_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


# Model
def convnet(x, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Conv Layer 1
    # filter size: 5x5, output: 32
    weights_1 = weights_variable([5, 5, 1, 32])
    biases_1 = bias_variable([32])
    conv1 = my_conv2d(x, weights_1, biases_1)

    # Pooling Layer 1
    pool1 = my_maxpool2d(conv1)

    # Conv Layer 2
    # filter size: 5x5, inputs: 32, outputs: 64
    weights_2 = weights_variable([5, 5, 32, 64])
    biases_2 = bias_variable([64])
    conv2 = my_conv2d(pool1, weights_2, biases_2)

    # Pooling Layer 2
    pool2 = my_maxpool2d(conv2)

    # Fully Connected Layer
    # image size: 7*7 (by 2 times max pooling) with 64 feature maps
    # output: 1024
    weight_fc1 = weights_variable([7*7*64, 1024])
    biases_fc1 = bias_variable([1024])
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    fc1 = tf.add(tf.matmul(pool2_flat, weight_fc1), biases_fc1)

    # Relu Layer
    fc1 = tf.nn.relu(fc1)

    # Dropout Layer
    fc1_drop = tf.nn.dropout(fc1, dropout)

    # Output Layer
    weight_out = weights_variable([1024, 10])
    biases_out = bias_variable([10])
    out = tf.add(tf.matmul(fc1_drop, weight_out), biases_out)
    return out


result = convnet(x, keep_prob)

# Loss function
loss_func = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(result, y))

# Optimizer
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss_func)

# Evaluation
correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variable
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(iterations):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout_rate})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_x, y: batch_y, keep_prob: dropout_rate})
            print("step %d, training accuracy %g" % (i, train_accuracy))

    testing = accuracy.eval(feed_dict={x: mnist.test.images,
                                       y: mnist.test.labels,
                                       keep_prob: dropout_rate})

    print("test accuracy: %g" % testing)
