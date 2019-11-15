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
        x = tf.layers.conv2d(encode, filters=256, kernel_size=(2, 2), padding='same', activation='relu')
        shape = tf.shape(x)
        h, w = shape[1], shape[2]
        x = tf.image.resize_nearest_neighbor(x, tf.stack([h*2, w*2]))
        
        # up sampling 2
        x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        shape = tf.shape(x)
        h, w = shape[1], shape[2]
        x = tf.image.resize_nearest_neighbor(x, tf.stack([h*2, w*2]))

        # up sampling 3
        x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        shape = tf.shape(x)
        h, w = shape[1], shape[2]
        x = tf.image.resize_nearest_neighbor(x, tf.stack([h*2, w*2]))

        # up sampling 4 (output)
        x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        x = tf.layers.conv2d(x, filters=3, kernel_size=(3, 3), padding='same', activation='sigmoid')

    return x

def adain(content_encode, style_encode, adain_layer, num_style, epsilon=1e-5):
    # get last cnn feature map of content and style for adain
    content_feature_map = content_encode[adain_layer]
    style_feature_map = style_encode[adain_layer]
    
    # compute statistics of content and style images
    c_mean, c_var = tf.nn.moments(content_feature_map, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)
    
    s_mean, s_var = tf.nn.moments(style_feature_map, axes=[1, 2], keep_dims=True)
    s_std = tf.sqrt(s_var + epsilon)
    
    # average statistics of style feature map
    s_mean = tf.reshape(s_mean, tf.concat([[-1, num_style], tf.shape(s_mean)[1:]], axis=0))
    s_std = tf.reshape(s_std, tf.concat([[-1, num_style], tf.shape(s_std)[1:]], axis=0))

    s_mean = tf.reduce_mean(s_mean, axis=1)
    s_std = tf.reduce_mean(s_std, axis=1)

    return s_std * (content_feature_map - c_mean) / c_std + s_mean



class VGG19(object):
    def __init__(self, w_path):
        self.w_path = w_path
        
    def vggnet(self, inputs):
        # load pre-trained vgg-19
        vgg = loadmat(self.w_path)
        vgg_layers = vgg['layers'][0]
        net = {}

        # keep cnn and pooling layer
        # tf.Variable(np.zeros([1, self.input_h, self.input_w, 3]), dtype=tf.float32)
        net['input'] = inputs
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
        # net['conv4_2'] = self.conv_relu(net['conv4_1'], self.get_wb(vgg_layers, 21))
        # net['conv4_3'] = self.conv_relu(net['conv4_2'], self.get_wb(vgg_layers, 23))
        # net['conv4_4'] = self.conv_relu(net['conv4_3'], self.get_wb(vgg_layers, 25))
        # net['pool4'] = self.pool(net['conv4_4'])
        
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

