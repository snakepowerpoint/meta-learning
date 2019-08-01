import pickle
import os

# keras
from keras.utils.np_utils import to_categorical

import numpy as np
import random

def load_dataset(data_dir):
    with open(data_dir, 'rb') as f:
        train_data = pickle.load(f)

    return train_data

def process_orig_datasets(orig_data, train_test_split=0.1):
    imgs = orig_data['image_data']
    labels = [no_images for _, no_images in sorted(orig_data['class_dict'].items())]

    # one-hot encoding label
    num_classes = len(labels)
    class_labels = np.repeat(np.arange(num_classes), [len(clss) for clss in labels])
    categorical_labels = to_categorical(class_labels)

    # permutate data and split data into training and test set
    for no_images in labels:
        np.random.shuffle(no_images)

    train_idx = [''] * len(labels)
    test_idx = [''] * len(labels)
    for i in range(len(train_idx)):
        num_label = len(labels[i])
        train_idx[i] = labels[i][0:int(num_label*(1-train_test_split))]
        test_idx[i] = labels[i][int(num_label*(1-train_test_split)):]

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    
    X_train, Y_train = imgs[train_idx], categorical_labels[train_idx]
    X_test, Y_test = imgs[test_idx], categorical_labels[test_idx]

    return X_train, Y_train, X_test, Y_test

def random_mini_batches(X_train, Y_train, mini_batch_size=32, seed=None):
    # shuffle

    # mini-batch

    return None

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def sample_task(orig_data, way=5, shot=5, query=15, batch_size=32):
    """
    Args:
        orig_data: 
        way:
        shot
        query:
        batch_size:

    Return:
        x_support: shape=[way*shot, 84, 84, 3], e.g. [25=5*5, 84, 84, 3]
        y_support: shape=[way*shot, 5], e.g. [25=5*5, 5]
        x_query: shape=[way*query, 84, 84, 3], e.g. [75=5*15, 84, 84, 3]
        y_query: shape=[way*query, 5], e.g. [75=5*15, 5]
    """
    
    # sample N=way class
    class_name = list(orig_data['class_dict'].keys())
    sampled_classes = random.sample(class_name, k=way)
    
    # sample K=shot example
    example_each_class = [orig_data['class_dict'][c] for c in sampled_classes]
    sampled_example = [random.sample(x, k=shot+query) for x in example_each_class]
    
    # image data (x)
    task_x = [orig_data['image_data'][x] for x in sampled_example]
    x_support = [x[:shot] for x in task_x]
    x_query = [x[shot:(shot+query)] for x in task_x]
    # concat and normalize RGB image data
    x_support = np.concatenate(x_support, axis=0) / 255
    x_query = np.concatenate(x_query, axis=0) / 255

    # label data (y)
    task_y = [np.repeat(x, repeats=shot+query) for x in range(way)]
    task_y = to_categorical(task_y)
    y_support = [y[:shot] for y in task_y]
    y_query = [y[shot:(shot+query)] for y in task_y]
    # concat
    y_support = np.concatenate(y_support, axis=0)
    y_query = np.concatenate(y_query, axis=0)
    
    return x_support, y_support, x_query, y_query
