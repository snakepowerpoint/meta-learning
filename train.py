# framework
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

# numpy
import numpy as np

# customerized 
from model import resnet18, weight_variable
from utils import load_dataset, process_orig_datasets, sample_task


# hyper-parameters
NUM_CLASS=5
EPOCHS=500

def compute_accuracy():
    return None

def loss(outputs, labels):
    with tf.name_scope('loss'):
        reduced_cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=outputs)
    return reduced_cross_entropy

def train(): # data, model, hyper-parameters, loss
    return None # loss, model?

def evaluate():
    return None

def main():
    # load data
    data_dir = 'data/mini-imagenet/mini-imagenet-cache-train.pkl'
    orig_data = load_dataset(data_dir)
    
    # establish model
    inputs = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])
    labels = tf.placeholder(tf.int32, shape=[None, 5])
    fc1, train_mode = resnet18(inputs=inputs)

    with tf.variable_scope('output'):
        # output layers
        w_output = tf.Variable(xavier_initializer()([512, NUM_CLASS]))
        #output = tf.matmul(fc1, w_output)

        # convert Euclidean space to angular space
        fc1_normalized = tf.nn.l2_normalize(fc1, axis=1)
        w_output_normalized = tf.nn.l2_normalize(w_output, axis=0)
        cos_dist = tf.matmul(fc1_normalized, w_output_normalized)
        #exp_cos_dist = tf.exp(cos_dist)
        #exp_cos_dist = tf.nn.softmax(exp_cos_dist)
    
    cross_entropy = loss(outputs=cos_dist, labels=labels)
    optimizer = tf.train.AdamOptimizer(1e-5)
    train_op = optimizer.minimize(cross_entropy)
    
    # training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i_epoch in range(EPOCHS):
            # sample a batch of tasks (episode)
            x_support, y_support, x_query, y_query = sample_task(orig_data)

            # shuffle data
            indices = np.arange(x_support.shape[0])
            np.random.shuffle(indices)
            x_support, y_support = x_support[indices], y_support[indices]

            sess.run(train_op, feed_dict={inputs: x_support,
                                          labels: y_support,
                                          train_mode: True})

            if i_epoch % 20 == 0:
                test_loss = sess.run(cross_entropy, feed_dict={inputs: x_query,
                                                               labels: y_query,
                                                               train_mode: False})
                print('step %d, test (query) cost %g' % (i_epoch, test_loss))


if __name__ == '__main__':
    main()
