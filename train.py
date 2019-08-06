# framework
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import tensorflow.contrib.slim as slim

# numpy
import numpy as np

# customerized 
from model import conv4, conv4_new, resnet12, weight_variable
from utils import load_dataset, process_orig_datasets, sample_task


# hyper-parameters
N_WAY=5
K_SHOT=5
K_QUERY=15
EPISODE = 2000

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

        # loss_i_class = 0
        # num_query_i_class = fc1_i_class.shape[0]
        # for i_query in range(num_query_i_class):
        #     i_class_i_query = fc1_i_class[i_query]
        #     loss_i_class += tf.norm(i_class_i_query - i_class_prototype)
        # loss_per_class.append(loss_i_class)
        
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

def train(): # data, model, hyper-parameters, loss
    return # loss, model?

def evaluate():
    return 

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main():
    # load data
    data_dir = 'data/mini-imagenet/mini-imagenet-cache-train.pkl'
    orig_data = load_dataset(data_dir)
    
    # inputs placeholder
    support = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])
    support_labels = tf.placeholder(tf.int32, shape=[None, 5])
    query = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])
    query_labels = tf.placeholder(tf.int32, shape=[None, 5])
    training = tf.placeholder(tf.bool, name='training')

    # establish model
    with tf.variable_scope('prototype') as scope:
        fc1_support = conv4_new(inputs=support, training=training) # [25, 512]
        scope.reuse_variables()
        fc1_query = conv4_new(inputs=query, training=False)  # [75, 512]

    # compute prototype from support set
    prototype = generate_prototype(fc1_support, support_labels)
    
    # compute intra-class loss
    intra_class_loss = compute_intra_loss(prototype, fc1_query, query_labels)

    # compute inter-class loss
    inter_class_loss = compute_inter_loss(prototype, fc1_query, query_labels)
    
    total_loss = tf.add(intra_class_loss, inter_class_loss)
    total_loss = tf.divide(total_loss, N_WAY * K_QUERY)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(total_loss)
    
    model_summary()

    # training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i_episode in range(EPISODE):
            # sample a batch of tasks
            x_support, y_support, x_query, y_query = sample_task(orig_data)

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
                train_loss = sess.run(total_loss, feed_dict={support: x_support,
                                                             support_labels: y_support,
                                                             query: x_query,
                                                             query_labels: y_query,
                                                             training: False})
                print('Episode %d, training cost %g' % (i_episode, train_loss))

                # test_loss = sess.run(total_loss, feed_dict={support: x_query,
                #                                             support_labels: y_query,
                #                                             query: x_query,
                #                                             query_labels: y_query,
                #                                             training: False})
                # print('Episode %d, test cost %g' % (i_episode, test_loss))


if __name__ == '__main__':
    main()
