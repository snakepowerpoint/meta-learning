import tensorflow as tf
from scipy.io import loadmat
import numpy as np



def encoder(inputs, reuse=False):    
    with tf.variable_scope("encoder", reuse=reuse):
        layers = {}
        # down sampling 1
        x = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        layers["conv1_1"] = x
        x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        layers["conv1_2"] = x
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))
        
        # down sampling 2
        x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        layers["conv2_1"] = x
        x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        layers["conv2_2"] = x
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))

        # down sampling 3 (encoded feature map)
        x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        layers["conv3_1"] = x
        x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        layers["conv3_2"] = x
        x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        layers["conv3_3"] = x
        x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), padding='same', activation='relu')
        layers["conv3_4"] = x
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))

    return x, layers

def decoder(encode):
    with tf.variable_scope("decoder"):
        # up sampling 1
        x = tf.layers.conv2d(encode, filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        shape = tf.shape(x)
        h, w = shape[1], shape[2]
        x = tf.image.resize_nearest_neighbor(x, tf.stack([h*2, w*2]))
        
        # up sampling 2
        x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        shape = tf.shape(x)
        h, w = shape[1], shape[2]
        x = tf.image.resize_nearest_neighbor(x, tf.stack([h*2, w*2]))

        # output
        x = tf.layers.conv2d(x, filters=3, kernel_size=(3, 3), padding='same', activation='relu')
        shape = tf.shape(x)
        h, w = shape[1], shape[2]
        x = tf.image.resize_nearest_neighbor(x, tf.stack([h*2, w*2]))

    return x

def adain(content_encode, style_encode, epsilon=1e-5):
    # compute mean and std 
    c_mean, c_var = tf.nn.moments(content_encode, axes=[1, 2], keep_dims=True)
    s_mean, s_var = tf.nn.moments(style_encode, axes=[1, 2], keep_dims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

    return s_std * (content_encode - c_mean) / c_std + s_mean

def conv4(inputs, reuse=False, training=True):
    with tf.variable_scope("conv4", reuse=reuse):
        # block 1
        x = conv_block(inputs=inputs, out_channels=64, ksize=3, block=1, training=training)
        x = spatial_attention(feature_map=x, out_channels=64, block=1)

        # block 2
        x = conv_block(x, 64, 3, 2, training)
        x = spatial_attention(feature_map=x, out_channels=64, block=2)

        # block 3
        x = conv_block(x, 64, 3, 3, training)
        x = spatial_attention(feature_map=x, out_channels=64, block=3)

        # block 4
        x = conv_block(x, 64, 3, 4, training)
        x = spatial_attention(feature_map=x, out_channels=64, block=4)

        # global average pooling
        avg_pool = tf.reduce_mean(x, axis=[1, 2], name='avg_pool', keepdims=True)
        flatten = tf.layers.flatten(avg_pool)

    return flatten

def conv_block(inputs, out_channels, ksize, block, training):
    block_name = 'block' + str(block)
    with tf.variable_scope(block_name):
        x = tf.layers.conv2d(inputs, out_channels, ksize, padding='same')
        x = tf.layers.batch_normalization(x, axis=-1, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    return x

def spatial_attention(feature_map, out_channels, block):
    block_name = 'block' + str(block) + '_spa_att'
    with tf.variable_scope(block_name):
        # down sampling 1
        x = tf.layers.conv2d(inputs=feature_map, filters=out_channels, kernel_size=3, padding='same')
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding='same')
        short_cut = tf.layers.conv2d(inputs=x, filters=out_channels, kernel_size=3, padding='same')

        # down sampling 2
        x = tf.layers.conv2d(inputs=x, filters=out_channels, kernel_size=3, padding='same')
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding='same')
        print(x.get_shape())

        # up sampling 1
        x = tf.layers.conv2d(inputs=x, filters=out_channels, kernel_size=3, padding='same')
        print(x.get_shape())
        x = tf.keras.layers.UpSampling2D(size=[2, 2])(x)
        print(short_cut.get_shape())
        x += short_cut

        # up sampling 2
        x = tf.layers.conv2d(inputs=x, filters=out_channels, kernel_size=3, padding='same')
        x = tf.keras.layers.UpSampling2D(size=[2, 2])(x)
        
        # residual attention
        x = (1 + feature_map) * x

    return x

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



class VGG19(object):
    def __init__(self, w_path, input_w, input_h):
        self.w_path = w_path
        self.input_h = input_h
        self.input_w = input_w
        self.model = self.vggnet()

    def vggnet(self):
        # load pre-trained vgg-19
        vgg = loadmat(self.w_path)
        vgg_layers = vgg['layers'][0]
        net = {}

        # keep cnn and pooling layer
        net['input'] = tf.Variable(np.zeros([1, self.input_h, self.input_w, 3]), dtype=tf.float32)
        net['conv1_1'] = self.conv_relu(net['input'], self.get_wb(vgg_layers, 0))
        net['conv1_2'] = self.conv_relu(net['conv1_1'], self.get_wb(vgg_layers, 2))
        net['pool1'] = self.pool(net['conv1_2'])
        
        net['conv2_1'] = self.conv_relu(net['pool1'], self.get_wb(vgg_layers, 5))
        net['conv2_2'] = self.conv_relu(net['conv2_1'], self.get_wb(vgg_layers, 7))
        net['pool2'] = self.pool(net['conv2_2'])
        
        net['conv3_1'] = self.conv_relu(net['pool2'], self.get_wb(vgg_layers, 10))
        net['conv3_2'] = self.conv_relu(net['conv3_1'], self.get_wb(vgg_layers, 12))
        net['conv3_3'] = self.conv_relu(net['conv3_2'], self.get_wb(vgg_layers, 14))
        net['conv3_4'] = self.conv_relu(net['conv3_3'], self.get_wb(vgg_layers, 16))
        net['pool3'] = self.pool(net['conv3_4'])
        
        net['conv4_1'] = self.conv_relu(net['pool3'], self.get_wb(vgg_layers, 19))
        net['conv4_2'] = self.conv_relu(net['conv4_1'], self.get_wb(vgg_layers, 21))
        net['conv4_3'] = self.conv_relu(net['conv4_2'], self.get_wb(vgg_layers, 23))
        net['conv4_4'] = self.conv_relu(net['conv4_3'], self.get_wb(vgg_layers, 25))
        net['pool4'] = self.pool(net['conv4_4'])
        
        # net['conv5_1'] = self.conv_relu(net['pool4'], self.get_wb(vgg_layers, 28))
        # net['conv5_2'] = self.conv_relu(net['conv5_1'], self.get_wb(vgg_layers, 30))
        # net['conv5_3'] = self.conv_relu(net['conv5_2'], self.get_wb(vgg_layers, 32))
        # net['conv5_4'] = self.conv_relu(net['conv5_3'], self.get_wb(vgg_layers, 34))
        # net['pool5'] = self.pool(net['conv5_4'])
        
        return net


    def conv_relu(self, inputs, wb):
        conv = tf.nn.conv2d(inputs, wb[0], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv + wb[1])
        return relu


    def pool(self, inputs):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def get_wb(self, layers, i):
        # load weights (set as constant) from pre-trained vgg model
        w = tf.constant(layers[i][0][0][0][0][0])
        bias = layers[i][0][0][0][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return w, b

