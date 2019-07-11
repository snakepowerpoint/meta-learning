import tensorflow as tf


def resnet18(inputs):  # num_class
    with tf.variable_scope("resnet"):
        training = tf.placeholder(tf.bool, name='training')

        # stage 1
        w_conv1 = weight_variable(shape=[7, 7, 3, 64])
        x = tf.nn.conv2d(inputs, w_conv1, strides=[1, 2, 2, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # stage 2
        x = convolutional_block(x, 3, [64, 64], 64, stride=1, stage=2, block=1, training=training)
        x = identity_block(x, 3, [64, 64], 64, stage=2, block=2, training=training)

        # stage 3
        x = convolutional_block(x, 3, [128, 128], 64, stride=2, stage=3, block=1, training=training)
        x = identity_block(x, 3, [128, 128], 128, stage=3, block=2, training=training)

        # stage 4
        x = convolutional_block(x, 3, [256, 256], 128, stride=2, stage=4, block=1, training=training)
        x = identity_block(x, 3, [256, 256], 256, stage=4, block=2, training=training)

        # stage 5
        x = convolutional_block(x, 3, [512, 512], 256, stride=2, stage=5, block=1, training=training)
        x = identity_block(x, 3, [512, 512], 512, stage=5, block=2, training=training)

        # global average pooling
        avg_pool = tf.reduce_mean(x, axis=[1, 2], name='avg_pool', keepdims=True)
        flatten = tf.layers.flatten(avg_pool)

        # fc layer
        fc1 = tf.layers.dense(flatten, units=512, activation=tf.nn.relu)
        #fc2 = tf.layers.dense(flatten, units=num_class, activation=tf.nn.softmax)
        
    return fc1, training
        
def convolutional_block(inputs, kernel_size, out_filter, in_filter, stride, stage, block, training):
    f1, f2 = out_filter
    block_name = 'res_conv' + str(stage) + "_" + str(block)
    with tf.variable_scope(block_name):
        short_cut = inputs

        # first
        w_conv1 = weight_variable(shape=[kernel_size, kernel_size, in_filter, f1])
        x = tf.nn.conv2d(inputs, w_conv1, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)

        # second
        w_conv2 = weight_variable(shape=[kernel_size, kernel_size, f1, f2])
        x = tf.nn.conv2d(x, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, training=training)

        # shortcut
        w_short = weight_variable(shape=[1, 1, in_filter, f2])
        short_cut = tf.nn.conv2d(short_cut, w_short, strides=[1, stride, stride, 1], padding='SAME')

        # add short cut
        add = tf.add(x, short_cut)
        add = tf.nn.relu(add)

    return add

def identity_block(inputs, kernel_size, out_filter, in_filter, stage, block, training):
    f1, f2 = out_filter
    block_name = 'res_conv' + str(stage) + "_" + str(block)
    with tf.variable_scope(block_name):
        short_cut = inputs

        # first
        w_conv1 = weight_variable(shape=[kernel_size, kernel_size, in_filter, f1])
        x = tf.nn.conv2d(inputs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)

        # second
        w_conv2 = weight_variable(shape=[kernel_size, kernel_size, f1, f2])
        x = tf.nn.conv2d(x, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, training=training)

        # add short cut
        add = tf.add(x, short_cut)
        add = tf.nn.relu(add)

    return add

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))
