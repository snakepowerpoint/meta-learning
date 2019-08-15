import tensorflow as tf


def conv4_new(inputs, reuse=False, training=True):
    with tf.variable_scope("conv4", reuse=reuse):
        # block 1
        x = conv_block(inputs=inputs, out_channels=64, ksize=3, block=1, training=training)
        # block 2
        x = conv_block(x, 64, 3, 2, training)
        # block 3
        x = conv_block(x, 64, 3, 3, training)
        # block 4
        x = conv_block(x, 64, 3, 4, training)

        # global average pooling
        avg_pool = tf.reduce_mean(x, axis=[1, 2], name='avg_pool', keepdims=True)
        flatten = tf.layers.flatten(avg_pool)

    return flatten

def conv_block(inputs, out_channels, ksize, block, training):
    block_name = 'block' + str(block)
    with tf.variable_scope(block_name):
        x = tf.layers.conv2d(inputs, out_channels, ksize, padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    return x

def conv4(inputs, training):
    """
    Args:
        inputs:
        training:
    Return:
        fc1: fully connected layer
    """
    
    with tf.variable_scope("conv4"):
        # block 1
        w_conv1 = weight_variable(shape=[3, 3, 3, 32])
        x = tf.nn.conv2d(inputs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # block 2
        w_conv2 = weight_variable(shape=[3, 3, 32, 32])
        x = tf.nn.conv2d(x, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # block 3
        w_conv3 = weight_variable(shape=[3, 3, 32, 32])
        x = tf.nn.conv2d(x, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # block 4
        w_conv4 = weight_variable(shape=[3, 3, 32, 32])
        x = tf.nn.conv2d(x, w_conv4, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # global average pooling
        avg_pool = tf.reduce_mean(x, axis=[1, 2], name='avg_pool', keepdims=True)
        flatten = tf.layers.flatten(avg_pool)

        # fc 1
        fc1 = tf.layers.dense(flatten, units=512, activation=tf.nn.relu)

    return fc1

def resnet12(inputs, training):  # num_class
    with tf.variable_scope("resnet"):
        # block 1
        x = convolutional_block(inputs, kernel_size=3, in_filter=3, out_filter=64, block=1, training=training)

        # block 2
        x = convolutional_block(inputs, 3, 64, 96, 2, training)

        # block 3
        x = convolutional_block(inputs, 3, 96, 128, 3, training)

        # block 4
        x = convolutional_block(inputs, 3, 128, 256, 4, training)
    
        # global average pooling
        w_conv1 = weight_variable(shape=[1, 1, 256, 2048])
        x = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')

        avg_pool = tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 1, 1, 1], padding='VALID')
        avg_pool = tf.nn.relu(avg_pool)
        avg_pool = tf.nn.dropout(avg_pool, keep_prob=0.9)

        w_conv2 = weight_variable(shape=[1, 1, 2048, 384])
        x = tf.nn.conv2d(x, w_conv2, strides=[1, 1, 1, 1], padding='SAME')

        avg_pool = tf.reduce_mean(x, axis=[1, 2], name='avg_pool', keepdims=True)
        flatten = tf.layers.flatten(avg_pool)

        # fc layer
        #fc1 = tf.layers.dense(flatten, units=512, activation=tf.nn.relu)
        #fc2 = tf.layers.dense(flatten, units=num_class, activation=tf.nn.softmax)
        
    return flatten
        
def convolutional_block(inputs, kernel_size, in_filter, out_filter, block, training):
    block_name = 'block' + str(block)
    with tf.variable_scope(block_name):
        short_cut = inputs

        # first
        w_conv1 = weight_variable(shape=[kernel_size, kernel_size, in_filter, out_filter])
        x = tf.nn.conv2d(inputs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # second
        w_conv2 = weight_variable(shape=[kernel_size, kernel_size, out_filter, out_filter])
        x = tf.nn.conv2d(x, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # third
        w_conv3 = weight_variable(shape=[kernel_size, kernel_size, out_filter, out_filter])
        x = tf.nn.conv2d(x, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # shortcut
        w_short = weight_variable(shape=[1, 1, in_filter, out_filter])
        short_cut = tf.nn.conv2d(short_cut, w_short, strides=[1, 1, 1, 1], padding='SAME')

        # add short cut
        add = tf.add(x, short_cut)
        add = tf.nn.max_pool(add, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        add = tf.nn.dropout(add, keep_prob=0.9)

    return add

def identity_block(inputs, kernel_size, out_filter, in_filter, stage, block, training):
    f1, f2 = out_filter
    block_name = 'res_conv' + str(stage) + "_" + str(block)
    with tf.variable_scope(block_name):
        short_cut = inputs

        # first
        w_conv1 = weight_variable(shape=[kernel_size, kernel_size, in_filter, f1])
        x = tf.nn.conv2d(inputs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.relu(x)

        # second
        w_conv2 = weight_variable(shape=[kernel_size, kernel_size, f1, f2])
        x = tf.nn.conv2d(x, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)

        # add short cut
        add = tf.add(x, short_cut)
        add = tf.nn.relu(add)

    return add

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(shape=shape, initializer=initializer)

def get_weight_variable(name, shape):
    """weight_variable generates a weight variable of a given shape."""
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name=name, shape=shape, initializer=initializer)
