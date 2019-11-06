# tensorflow
import tensorflow as tf
import tensorflow.contrib.slim as slim

# numpy
import numpy as np

# io
import pickle as pkl

# customerized 
from model import decoder, adain, VGG19
from losses import compute_content_loss, compute_style_loss
from utils import resize_image, generate_few_shot_style

# miscellaneous
import gc



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
    mm_train, mm_train_y = mm['train'], mm['train_label']; del mm

    #  keep numbers 0-4 in MNIST as content, and numbers 5-9 in MNIST-M as style
    content_image = m_train[m_train_y < 5, ...]
    content_image_y = m_train_y[m_train_y < 5]
    content_image = np.repeat(content_image[..., np.newaxis], 3, axis=-1)
    
    test_content_image = m_train[m_train_y >= 5, ...]; del m_train
    test_content_image_y = m_train_y[m_train_y >= 5]; del m_train_y
    test_content_image = np.repeat(test_content_image[..., np.newaxis], 3, axis=-1)
    
    style_image = mm_train[mm_train_y >= 5, ...]; del mm_train
    style_image_y = mm_train_y[mm_train_y >= 5]; del mm_train_y
    style_image, style_image_y = generate_few_shot_style(style_image, style_image_y, num_sample=5)
    
    # resize
    content_image = resize_image(content_image, size=(32, 32))
    test_content_image = resize_image(test_content_image, size=(32, 32))
    style_image = resize_image(style_image, size=(32, 32))    

    # # use all train data in MNIST as content image, and all train data in MNIST-M as style image
    # content_image = np.repeat(m_train[..., np.newaxis], 3, axis=-1); del m_train
    
    # test_content_image = np.repeat(m_test[..., np.newaxis], 3, axis=-1); del m_test
    
    # # resize    
    # content_image = resize_image(content_image, size=(32, 32))
    # test_content_image = resize_image(test_content_image, size=(32, 32))
    # style_image = resize_image(mm_train, size=(32, 32)); del mm_train
    
    # normalize
    content_image = content_image / 255
    style_image = style_image / 255
    test_content_image = test_content_image / 255

    gc.collect()


    ## prepare model
    model_path = "model/imagenet-vgg-verydeep-19.mat"
    encoder = VGG19(model_path)

    # inputs placeholder
    c_img = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    s_img = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    
    # establish model
    c_encode = encoder.vggnet(c_img)
    s_encode = encoder.vggnet(s_img)

    c_adain_encode = adain(c_encode['conv4_1'], s_encode['conv4_1'])
    styled_img = decoder(c_adain_encode)
    styled_encode = encoder.vggnet(styled_img)

    # loss
    content_loss = compute_content_loss(styled_encode['conv4_1'], c_adain_encode)
    style_loss = compute_style_loss(styled_encode, s_encode)
    total_loss = content_loss + 0.01 * style_loss
    
    # optimizer 
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(total_loss)
    
    model_summary()


    ## training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter("logs/train_l1e3/", sess.graph)
        file_writer_test = tf.summary.FileWriter("logs/test_l1e3/", sess.graph)

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
        num_episode = 3000

        for i_episode in range(num_episode):
            # shuffle data
            np.random.shuffle(content_image)
            np.random.shuffle(style_image)

            for i_batch in range(num_batch):
                # get a batch of content
                c_image = content_image[i_batch*batch_size: (i_batch+1)*batch_size, ...]
                
                # random sample a batch of style
                idx = np.random.choice(style_image.shape[0], batch_size, replace=False)
                s_image = style_image[idx, ...]
                
                # training                 
                _, train_loss = sess.run([train_op, total_loss], feed_dict={
                    c_img: c_image,
                    s_img: s_image
                })

                if i_batch % 100 == 0:
                    # evaluation on test content image
                    np.random.shuffle(test_content_image)
                    
                    test_c_image = test_content_image[:10, ...]
                    test_s_image = style_image[:10, ...]

                    summary_train, train_loss = sess.run([merged, total_loss], feed_dict={
                        c_img: c_image,
                        s_img: s_image
                    })

                    summary_test, test_loss = sess.run([merged, total_loss], feed_dict={
                        c_img: test_c_image,
                        s_img: test_s_image
                    })

                    # log all variables
                    num_iter = i_episode * num_batch + i_batch
                    file_writer_train.add_summary(summary_train, global_step=num_iter)
                    file_writer_test.add_summary(summary_test, global_step=num_iter)

                    print('Episode: %d, batch: %d, training cost: %g, test cost: %g' %
                          (i_episode, i_batch, train_loss, test_loss))

        file_writer_train.close()
        file_writer_test.close()



if __name__ == '__main__':
    main()
