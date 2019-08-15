# framework
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import tensorflow.contrib.slim as slim

# numpy
import numpy as np

# customerized 
from model import conv4, conv4_new, resnet12
from utils import load_dataset, sample_task, generate_evaluation_data


# hyper-parameters
N_WAY=5
K_SHOT=1
K_QUERY=15
EPISODE = 30000

def compute_accuracy():
    return None

def loss(labels, outputs):
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=outputs)
    return cross_entropy

def generate_prototype(fc1_support, support_labels):
    num_class = support_labels.shape[1]

    mean_vector = []
    for i_class in range(num_class):
        i_class_mask = tf.equal(support_labels[:, i_class], 1)
        fc1_i_class = tf.boolean_mask(fc1_support, i_class_mask)
        mean_vector.append(tf.reduce_mean(fc1_i_class, axis=0))
    mean_vector = tf.stack(mean_vector)

    return mean_vector

def compute_intra_loss(prototype, fc1_query, query_labels):
    '''
    Compute intra-class loss

    Args
        prototype: [5, 512]
        fc1_query: [75, 512]
        query_labels: [75, 5]
        
    Return
        scalar
    '''
    num_class = prototype.shape[0]

    loss_per_class = []
    for i_class in range(num_class):
        i_class_prototype = prototype[i_class]  # i-th prototype

        i_class_mask = tf.equal(query_labels[:, i_class], 1)
        fc1_i_class = tf.boolean_mask(fc1_query, i_class_mask)  # all queries for i-th class

        i_class_query_norm = tf.norm(fc1_i_class - i_class_prototype, axis=1)
        loss_per_class.append(tf.reduce_sum(i_class_query_norm))        
        
    intra_class_loss = tf.reduce_sum(loss_per_class)

    return intra_class_loss

def compute_inter_loss(prototype, fc1_query, query_labels):
    '''
    Compute inter-class loss

    Args
        prototype: [5, 512]
        fc1_query: [75, 512]
        query_labels: [75, 5]

    Return
        scalar
    '''
    num_class = prototype.shape[0]

    loss_per_class = []
    for i_class in range(num_class):
        i_class_prototype = prototype[i_class]  # i-th prototype

        # select all queries that are not i-class
        i_class_mask = tf.equal(query_labels[:, i_class], 0)
        fc1_not_i_class = tf.boolean_mask(fc1_query, i_class_mask)  

        i_class_other_query_norm = tf.norm(fc1_not_i_class - i_class_prototype, axis=1)
        i_class_loss = tf.exp(tf.negative(i_class_other_query_norm))  # exp(-d(prototype, y))
        i_class_loss = tf.log(tf.reduce_sum(i_class_loss))  # log(sum(loss_per_class))
        loss_per_class.append(i_class_loss)

    inter_class_loss = tf.reduce_sum(loss_per_class)

    return inter_class_loss

def compute_distance(fc1_query, prototype):
    '''
    Compute distance

    Args
        fc1_query: [75, 512]
        prototype: [5, 512]

    Return
        scalar
    '''

    M, D = tf.shape(fc1_query)[0], tf.shape(fc1_query)[1]
    N = tf.shape(prototype)[0]
    fc1_query = tf.tile(tf.expand_dims(fc1_query, axis=1), (1, N, 1))
    prototype = tf.tile(tf.expand_dims(prototype, axis=0), (M, 1, 1))
    dist = tf.reduce_mean(tf.square(fc1_query - prototype), axis=-1)
    
    return dist

def compute_acc(prediction, one_hot_labels):
    labels = tf.argmax(one_hot_labels, axis=1)
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(prediction, axis=-1), labels)))

    return acc

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main():
    # load training data
    data_dir = 'data/mini-imagenet/mini-imagenet-cache-train.pkl'
    orig_data = load_dataset(data_dir)

    # load test data
    data_dir = 'data/mini-imagenet/mini-imagenet-cache-test.pkl'
    test_orig_data = load_dataset(data_dir)
    test_data = generate_evaluation_data(test_orig_data, way=N_WAY, shot=K_SHOT, query=K_QUERY)
    
    # inputs placeholder
    support = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])
    support_labels = tf.placeholder(tf.float32, shape=[None, 5])
    query = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])
    query_labels = tf.placeholder(tf.float32, shape=[None, 5])
    training = tf.placeholder(tf.bool, name='training')

    # establish model
    fc1_support = conv4_new(inputs=support, training=training) # [25, 512]
    fc1_query = conv4_new(inputs=query, reuse=True, training=False)  # [75, 512]

    # compute prototype from support set
    prototype = generate_prototype(fc1_support, support_labels)
    
    # compute distance
    dists = compute_distance(fc1_query, prototype)
    log_p_y = tf.nn.log_softmax(-dists)
    loss_per_query = tf.reduce_sum(tf.multiply(query_labels, log_p_y), axis=-1)
    ce_loss = -tf.reduce_mean(loss_per_query)
    acc = compute_acc(log_p_y, query_labels)

    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(ce_loss)
    
    model_summary()

    # training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i_episode in range(EPISODE):
            # sample a batch of tasks
            x_support, y_support, x_query, y_query = sample_task(orig_data, way=N_WAY, shot=K_SHOT, query=K_QUERY)

            # shuffle data 
            indices = np.arange(x_support.shape[0])
            np.random.shuffle(indices)
            x_support, y_support = x_support[indices], y_support[indices]

            sess.run(train_op, feed_dict={support: x_support,
                                          support_labels: y_support,
                                          query: x_query,
                                          query_labels: y_query,
                                          training: True})

            if i_episode % 200 == 0:
                train_loss, accuracy = sess.run([ce_loss, acc], feed_dict={support: x_support,
                                                                           support_labels: y_support,
                                                                           query: x_query,
                                                                           query_labels: y_query,
                                                                           training: False})
                print('Episode: %d, training cost: %g, acc: %g' % (i_episode, train_loss, accuracy))
                
                idx = np.random.choice(range(len(test_data)))
                x_support, y_support, x_query, y_query = test_data[idx]
                test_loss, accuracy = sess.run([ce_loss, acc], feed_dict={support: x_query,
                                                                          support_labels: y_query,
                                                                          query: x_query,
                                                                          query_labels: y_query,
                                                                          training: False})
                print('Episode: %d, test cost: %g, acc: %g' % (i_episode, test_loss, accuracy))
            
        # evaluation for all testing episodes
        test_loss_all = []
        acc_all = []
        num_episode = len(test_data)
        for idx in range(num_episode):
            x_support, y_support, x_query, y_query = test_data[idx]
            test_loss, accuracy = sess.run([ce_loss, acc], feed_dict={support: x_query,
                                                                      support_labels: y_query,
                                                                      query: x_query,
                                                                      query_labels: y_query,
                                                                      training: False})
            test_loss_all.append(test_loss)
            acc_all.append(accuracy)
        print('Final test cost: %g, acc: %g, acc_std: %g' %
              (np.mean(test_loss_all), np.mean(acc_all), np.std(acc_all)))



if __name__ == '__main__':
    main()
