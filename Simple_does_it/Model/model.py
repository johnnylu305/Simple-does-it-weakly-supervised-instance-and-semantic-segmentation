import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import random
import tqdm
from tensorflow.python.training import saver

BASEDIR = os.path.join(os.path.dirname(__file__), '..')

sys.path.insert(0, BASEDIR)

from Dataset.load import Load
from Parser_.parser import model_parser
from Dataset.save_result import Save
from Postprocess.dense_CRF import dense_CRF

args = model_parser()

# parameter for Loading
DATASET = args.dataset
SET_NAME = args.set_name
LABEL_DIR_NAME = args.label_dir_name
IMG_DIR_NAME = args.img_dir_name

# dataset
# classes for segmentation
# default: 21
CLASS = args.classes
# training set size
# default: get from loading data
TRAIN_SIZE = None
# testing set size
# default: get from loading data
TEST_SIZE = None


# output format
SPACE = 15

# tqdm parameter
UNIT_SCALE = True
BAR_FORMAT = '{}{}{}'.format('{l_bar}', '{bar}', '| {n_fmt}/{total_fmt}')

# hyperparameter
# batch size
# default: 3 
BATCH_SIZE = args.batch_size
# epoch
# default: 30000
EPOCH = args.epoch
# learning rate
# defalut: 0.001
LR = args.learning_rate
# momentum for optimizer
# default: 0.9
MOMENTUM = tf.Variable(args.momentum)
# probability for dropout
# default: 1
KEEP_PROB = args.keep_prob
# training or testing
# default: False
IS_TRAIN = args.is_train
# iteration
# ITER = TRAIN_SIZE/BATCH_SIZE
ITER = None
# widht and height after resize
# get from loading data
WIDTH = args.width
HEIGHT = args.height

# saving and restore weight
# VGG_16 
VGG16_CKPT_PATH = BASEDIR + "/Model/models/vgg_16.ckpt"
# saving weight each SAVE_STEP 
# default: 10
SAVE_STEP = args.save_step
# resore weights number
RESTORE_TARGET = args.restore_target
# restore weights path
RESTORE_CKPT_PATH = BASEDIR + "/Model/models/model_"+ RESTORE_TARGET + ".ckpt"

# location for saving results
PRED_DIR_PATH = DATASET + '/' + args.pred_dir_name
PAIR_DIR_PATH = DATASET + '/' + args.pair_dir_name
CRF_DIR_PATH = DATASET + '/' + args.crf_dir_name
CRF_PAIR_DIR_PATH = DATASET + '/' + args.crf_pair_dir_name

# define placeholder 
xp = tf.placeholder(tf.float32, shape = (None, None, None, 3))
yp = tf.placeholder(tf.int32, shape = (None, None, None, 1))

# build convolution layer for deeplab
def build_conv(input_, shape, name, strides = [1, 1, 1, 1], padding = 'SAME', activation = True, c_name = 'PRETRAIN_VGG16', holes = None):
    # tf.AUTO_REUSE for using exist variable
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        # define initializer for weights and biases
        w_initializer = tf.contrib.layers.xavier_initializer()
        b_initializer = tf.zeros_initializer()
        # define variable for weights and biases
        biases = tf.get_variable(initializer = b_initializer, shape = shape[-1], name = 'biases', collections = [c_name, tf.GraphKeys.GLOBAL_VARIABLES])
        kernel = tf.get_variable(initializer = w_initializer, shape = shape, name = 'weights', collections = [c_name, tf.GraphKeys.GLOBAL_VARIABLES])
        # convolution
        if not holes: 
            layer = tf.nn.conv2d(input = input_, filter = kernel, strides = strides, padding = padding)
        else:
            layer = tf.nn.atrous_conv2d(value = input_, filters = kernel, rate = holes, padding = padding)
        # add biases
        layer = tf.nn.bias_add(layer, biases)
        # use activation or not
        if activation:
            layer = tf.nn.relu(batch_norm(layer))
    
    return layer

# define network
def network():
    # get input from placeholder
    x = xp
    y = yp
    BATCH_SIZE = tf.shape(x)[0]
    WIDTH = tf.shape(x)[1]
    HEIGHT = tf.shape(x)[2]
    # DeepLab-LargeFOV
    with tf.variable_scope('vgg_16'): 
        with tf.variable_scope('conv1'):
            layer1 = build_conv(x, [3, 3, 3, 64], 'conv1_1')
            layer2 = build_conv(layer1, [3, 3, 64, 64], 'conv1_2')
            pool1 = tf.nn.max_pool(value = layer2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool1')
        
        with tf.variable_scope('conv2'):       
            layer3 = build_conv(pool1, [3, 3, 64, 128], 'conv2_1')
            layer4 = build_conv(layer3, [3, 3, 128, 128], 'conv2_2')
            pool2 = tf.nn.max_pool(value = layer4, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool2')
        
        with tf.variable_scope('conv3'):
            layer5 = build_conv(pool2, [3, 3, 128, 256], 'conv3_1') 
            layer6 = build_conv(layer5, [3, 3, 256, 256], 'conv3_2')
            layer7 = build_conv(layer6, [3, 3, 256, 256], 'conv3_3')
            pool3 = tf.nn.max_pool(value = layer7, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool3')
        
        with tf.variable_scope('conv4'):
            layer8 = build_conv(pool3, [3, 3, 256, 512], 'conv4_1')
            layer9 = build_conv(layer8, [3, 3, 512, 512], 'conv4_2')
            layer10 = build_conv(layer9, [3, 3, 512, 512], 'conv4_3')
            pool4 = tf.nn.max_pool(value = layer10, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME', name = 'pool4')
        
        with tf.variable_scope('conv5'):
            layer11 = build_conv(pool4, [3, 3, 512, 512], 'conv5_1',  c_name = 'UNPRETRAIN', holes = 2)
            layer12 = build_conv(layer11, [3, 3, 512, 512], 'conv5_2',  c_name = 'UNPRETRAIN', holes = 2)
            layer13 = build_conv(layer12, [3, 3, 512, 512], 'conv5_3',  c_name = 'UNPRETRAIN', holes = 2)
            pool5 = tf.nn.max_pool(value = layer13, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME', name = 'pool5')
            pool5_1 = tf.nn.avg_pool(value = pool5, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME', name = 'pool5_1')
            
        layer14 = build_conv(pool5_1, [3, 3, 512, 1024], 'fc6', padding = 'SAME',  c_name = 'UNPRETRAIN', holes = 12)
        dropout6 = tf.nn.dropout(layer14, keep_prob = KEEP_PROB, name = 'dropout6')
        
        layer15 = build_conv(dropout6, [1, 1, 1024, 1024], 'fc7', padding = 'VALID', c_name = 'UNPRETRAIN')
        dropout7 = tf.nn.dropout(layer15, keep_prob = KEEP_PROB, name = 'dropout7')

        layer16 = build_conv(dropout7, [1, 1, 1024, CLASS], 'fc8', padding = 'VALID', activation = False, c_name = 'UNPRETRAIN')
        
        predictions = layer16

    # to one-hot    
    y = tf.reshape(y, shape = [BATCH_SIZE, -1])
    y = tf.one_hot(y, depth = CLASS)
    y = tf.reshape(y, shape = [-1, CLASS])
    # resize predictions for cross entropy
    predictions = tf.image.resize_bilinear(predictions, [WIDTH, HEIGHT])
    predictions = tf.reshape(predictions, [-1, CLASS])
    prob_prediction = tf.reshape(tf.nn.softmax(predictions), [BATCH_SIZE, WIDTH, HEIGHT, CLASS])

    # define loss function
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predictions, labels = y))
        tf.summary.scalar('loss', loss)
    
    # define optimizer
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate = LR,momentum = MOMENTUM).minimize(loss)

    # resize to image format
    predictions = tf.argmax(predictions, axis = 1)
    predictions = tf.reshape(predictions, [BATCH_SIZE, WIDTH, HEIGHT, 1])
    y = tf.argmax(y, axis = 1)
    y = tf.reshape(y, shape = [BATCH_SIZE, WIDTH, HEIGHT, 1])
    
    return loss, optimizer, predictions, y, prob_prediction

# batch norm
def batch_norm(data):
    axis = list(range(len(data.get_shape()) - 1))
    fc_mean, fc_var = tf.nn.moments(data, axes = axis)
    dimension = data.get_shape().as_list()[-1]
    shift = tf.Variable(tf.zeros([dimension]))
    scale = tf.Variable(tf.ones([dimension]))     
    epsilon = 0.001
    return tf.nn.batch_normalization(data, fc_mean, fc_var, shift, scale, epsilon)

# shuffle data
def shuffle_unison(x, y):
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

# augmentation
def augmentation(img, label):
    temp1 = np.pad(img, ((0, 0), (15, 15), (15, 15), (0, 0)), 'constant')
    temp2 = np.pad(label, ((0, 0), (15, 15), (15, 15), (0, 0)), 'constant')
    for i in range(img.shape[0]):
        # translation
        shift1 = random.randint(0, 30) if random.randint(0, 1) else 15
        shift2 = random.randint(0, 30) if random.randint(0, 1) else 15
        img[i] = temp1[i][shift1:WIDTH + shift1, shift2:HEIGHT + shift2][:]
        label[i] = temp2[i][shift1:WIDTH + shift1, shift2:HEIGHT + shift2][:]
        # flip
        if random.randint(0,1) == 0:
            img[i] = np.flip(img[i], 1)
            label[i] = np.flip(label[i], 1)
    return img, label

# training 
def train_network(x_train, y_train):
    with tf.Session() as sess:
        # get network
        loss, optimizer, predictions, y, prob_predictions = network()
        # setup tensorboard
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(BASEDIR + "/Model/Logs/", sess.graph)
        if RESTORE_TARGET == '0':
            # setup saver and restorer
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 500)
            restorer = tf.train.Saver(tf.get_collection('PRETRAIN_VGG16'))
            # initial MOMENTUM and batch normalization weights
            sess.run(tf.global_variables_initializer())       
            # load weight for untrainable variables
            restorer.restore(sess, VGG16_CKPT_PATH)
            # initial trainable variables
            init = tf.initialize_variables(tf.get_collection('UNPRETRAIN'))
            sess.run(init)
        else:
            # setup saver
            saver = tf.train.Saver(tf.global_variables())
            # load weight
            saver.restore(sess, RESTORE_CKPT_PATH)
        # training
        for i in range(EPOCH):
            print ('{:{}}: {}'.format('Epoch', SPACE, i))
            # shuffle data
            shuffle_unison(x_train, y_train)
            # augmentation
            x_train_, y_train_ = augmentation(copy.deepcopy(x_train), copy.deepcopy(y_train))
            # split for batch
            x_train_ = np.array_split(x_train_, ITER)
            y_train_ = np.array_split(y_train_, ITER)
            # save weight
            if i%SAVE_STEP==0:
                saver.save(sess, BASEDIR + "/Model/models/model_" + str(i + int(RESTORE_TARGET)) + ".ckpt")
                # learning rate decay
            if i % 100 == 0:
                global LR
                LR = LR/10
            for j in tqdm.tqdm(range(ITER), desc = '{:{}}'.format('Epoch' + str(i), SPACE), unit_scale = UNIT_SCALE, bar_format = BAR_FORMAT):
                # check empty or not
                if x_train_[j].size:
                    summary, optimizer_, loss_ = sess.run([merged, optimizer, loss], feed_dict={xp: x_train_[j], yp: y_train_[j]})
                    writer.add_summary(summary, i * ITER + j)
            print ('{:{}}: {}'.format('Final Loss', SPACE, loss_))
        writer.close()
# testing
def test_network(x_test, img_names):
    with tf.Session() as sess:
        # get network
        loss, optimizer, predictions, y, prob_predictions = network()
        # setup restorer
        restorer = tf.train.Saver(tf.global_variables())
        # load weight
        restorer.restore(sess, RESTORE_CKPT_PATH)
        for i in tqdm.tqdm(range(TEST_SIZE), desc = '{:{}}'.format('Test and save', SPACE), unit_scale = UNIT_SCALE, bar_format = BAR_FORMAT):
            predictions_, prob_predictions_ = sess.run([predictions, prob_predictions], feed_dict={xp: [x_test[i]]})
            save_ = Save(copy.deepcopy(x_test[i]), np.squeeze(predictions_), img_names[i], PRED_DIR_PATH, PAIR_DIR_PATH, CLASS)
            save_.save()

            dense_CRF_ = dense_CRF(x_test[i], prob_predictions_[0])
            crf_mask = dense_CRF_.run_dense_CRF()
            save_ = Save(copy.deepcopy(x_test[i]), crf_mask, img_names[i], CRF_DIR_PATH, CRF_PAIR_DIR_PATH, CLASS)
            save_.save()
def main():
    global WIDTH
    global HEIGHT
    global TRAIN_SIZE
    global KEEP_PROB
    global TEST_SIZE
    global ITER
    global BATCH_SIZE

    if IS_TRAIN:
        # load training data from VOC12 dataset
        dataset = Load(IS_TRAIN, DATASET, SET_NAME, LABEL_DIR_NAME, IMG_DIR_NAME, WIDTH, HEIGHT)
        x_train, y_train = dataset.load_data()
        # set training set size
        TRAIN_SIZE = len(x_train)
        # get iteration
        ITER = math.ceil(TRAIN_SIZE / BATCH_SIZE)
        # get widht and height
        WIDTH = x_train[0].shape[0]
        HEIGHT = x_train[0].shape[1]
        # train network
        train_network(x_train, y_train)

    else:
        # load val data from VOC12 dataset
        dataset = Load(IS_TRAIN, DATASET, SET_NAME, LABEL_DIR_NAME, IMG_DIR_NAME, WIDTH, HEIGHT)
        x_test, img_names = dataset.load_data()
        # set testing set size
        TEST_SIZE = len(x_test)
        # close dropout
        KEEP_PROB = 1
        # set batch size
        BATCH_SIZE = 1
        # test network
        test_network(x_test, img_names)

if __name__=='__main__':
    main()

