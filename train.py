# tensorflow
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import tensorflow.contrib.slim as slim

# numpy
import numpy as np

# customerized 
from model import encoder, decoder, adain
from losses import compute_content_loss, compute_style_loss
from utils import resize_image, generate_few_shot_style

import pickle as pkl
import cv2


# hyper-parameters
N_WAY=5
K_SHOT=1
K_QUERY=15
EPISODE=30000

def compute_accuracy():
    return None

def compute_loss(labels, outputs):
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
    ## prepare data
    # load MNIST and MNIST-M
    (m_train, m_train_y), (m_test, m_test_y) = tf.keras.datasets.mnist.load_data()
    mm = pkl.load(open('data/mnistm_data.pkl', 'rb'))
    mm_train, mm_train_y = mm['train'], mm['train_label']

    # #  keep numbers 0-4 in MNIST as content, and numbers 5-9 in MNIST-M as style
    # content_image = m_train[m_train_y < 5, ...]
    # content_image_y = m_train_y[m_train_y < 5]
    # content_image = resize_image(content_image, size=(32, 32))
    # content_image = np.repeat(content_image[..., np.newaxis], 3, axis=-1)
    # test_content_image = m_train[m_train_y >= 5, ...]
    
    # style_image = mm_train[mm_train_y >= 5, ...]
    # style_image_y = mm_train[mm_train_y >= 5]
    # style_image, style_image_y = generate_few_shot_style(style_image, style_image_y, num_sample=5)
    # style_image = resize_image(style_image, size=(32, 32))

    # use all train data in MNIST as content image, and all train data in MNIST-M as style image
    content_image = resize_image(m_train, size=(32, 32))
    content_image = np.repeat(content_image[..., np.newaxis], 3, axis=-1)
    test_content_image = m_test

    style_image = resize_image(mm_train, size=(32, 32))

    
    ## prepare model
    # inputs placeholder
    c_img = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    s_img = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    
    # establish model
    c_encode, _ = encoder(c_img)
    s_encode, s_layers = encoder(s_img, reuse=True)

    c_adain_encode = adain(c_encode, s_encode)
    styled_img = decoder(c_adain_encode)
    styled_encode, styled_layers = encoder(styled_img, reuse=True)

    # loss
    content_loss = compute_content_loss(styled_encode, c_adain_encode)
    style_loss = compute_style_loss(styled_layers, s_layers)
    total_loss = content_loss + 0.01 * style_loss

    # optimizer 
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(total_loss)
    
    model_summary()

    ## training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Creates a file writer for the log directory.
        logdir = "logs/"
        file_writer = tf.summary.FileWriter(logdir, sess.graph)

        # store variables
        tf.summary.image("Content image", c_img, max_outputs=10)
        tf.summary.image("Style image", s_img, max_outputs=10)
        tf.summary.image("Styled image", styled_img, max_outputs=10)
        tf.summary.scalar("Content loss", content_loss)
        tf.summary.scalar("Style loss", style_loss)
        tf.summary.scalar("Total loss", total_loss)
        merged = tf.summary.merge_all()

        sess.run(init)

        # total number of data
        num_data = content_image.shape[0]
        batch_size = 8
        num_batch = num_data // batch_size

        for i_episode in range(EPISODE):
            # shuffle data
            np.random.shuffle(content_image)
            np.random.shuffle(style_image)

            for i_batch in range(num_batch):
                # get a batch of content
                c_image = content_image[i_batch*batch_size: (i_batch+1)*batch_size, ...]
                c_image = c_image / 255

                # random sample a batch of style
                idx = np.random.choice(style_image.shape[0], batch_size, replace=False)
                s_image = style_image[idx, ...]
                s_image = s_image / 255

                # training                 
                _, train_loss = sess.run([train_op, total_loss], feed_dict={
                    c_img: c_image,
                    s_img: s_image
                })

                if i_batch % 100 == 0:
                    # evaluation on test content image
                    np.random.shuffle(test_content_image)
                    
                    test_c_image = test_content_image[:10, ...]
                    test_c_image = resize_image(test_c_image, size=(32, 32))
                    test_c_image = np.repeat(test_c_image[..., np.newaxis], 3, axis=-1)
                    test_c_image = test_c_image / 255

                    test_s_image = style_image[:10, ...] / 255

                    summary, test_loss = sess.run([merged, total_loss], feed_dict={
                        c_img: test_c_image,
                        s_img: test_s_image
                    })

                    # log all variables
                    #num_iter = i_episode * num_batch + i_batch
                    file_writer.add_summary(summary, global_step=i_episode * num_batch + i_batch)

                    print('Episode: %d, batch: %d, training cost: %g, test cost: %g' %
                          (i_episode, i_batch, train_loss, test_loss))

                    
        file_writer.close()

            # if i_episode % 100 == 0:
            #     # evaluation on test data
            #     content_image = m_train[m_train_y < 5, ...]
            #     content_image_y = m_train_y[m_train_y < 5]
            #     content_image = resize_image(content_image, size=(32, 32))
            #     content_image = np.expand_dims(content_image, axis=-1)

            #     num_test = len(test_data)
            #     test_losses = []
            #     for idx in range(num_test):
            #         test_c_img, test_s_img = test_data[idx]
            #         test_loss = sess.run([total_loss], feed_dict={
            #             c_img: test_c_img,
            #             s_img: test_s_img
            #         })
            #         test_losses.append(test_loss)

            #     print('Test cost: %g' % (np.mean(test_losses)))


if __name__ == '__main__':
    main()
