# framework
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import tensorflow.contrib.slim as slim

# numpy
import numpy as np

# customerized 
from model import conv4, resnet12, weight_variable
from utils import load_dataset, process_orig_datasets, sample_task


# hyper-parameters
NUM_CLASS=5
EPISODE = 50000

def compute_accuracy():
    return None

def loss(labels, outputs):
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=outputs)
    return cross_entropy

def train(): # data, model, hyper-parameters, loss
    return None # loss, model?

def evaluate():
    return None

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main():
    # load data
    data_dir = 'data/mini-imagenet/mini-imagenet-cache-train.pkl'
    orig_data = load_dataset(data_dir)
    
    # establish model
    support = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])
    query = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])
    labels = tf.placeholder(tf.int32, shape=[None, 5])
    training = tf.placeholder(tf.bool, name='training')

    fc1_support = conv4(inputs=support, training=training) # [25, 512]
    fc1_query = conv4(inputs=query, training=training) # [75, 512]

    mean_vector = tf.zeros(shape=[5, 512])
    for i in range(5):
        mean_vector[i] = tf.math.reduce_mean(fc1_support[i*5:(i+1)*5], axis=0)
    
    ## input query set


    
    cross_entropy = loss(labels=labels, outputs=output)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(cross_entropy)
    
    model_summary()

    # training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i_episode in range(EPISODE):
            # sample a batch of tasks
            x_support, y_support, x_query, y_query = sample_task(orig_data)

            # shuffle data (need to be modified)
            indices = np.arange(x_support.shape[0])
            np.random.shuffle(indices)
            x_support, y_support = x_support[indices], y_support[indices]

            sess.run(train_op, feed_dict={inputs: x_support,
                                          labels: y_support,
                                          training: True})

            if i_episode % 1000 == 0:
                train_loss = sess.run(cross_entropy, feed_dict={inputs: x_support,
                                                                labels: y_support,
                                                                training: False})
                print('Episode %d, train (support) cost %g' % (i_episode, train_loss))

                test_loss = sess.run(cross_entropy, feed_dict={inputs: x_query,
                                                               labels: y_query,
                                                               training: False})
                print('Episode %d, test (query) cost %g' % (i_episode, test_loss))


if __name__ == '__main__':
    main()
