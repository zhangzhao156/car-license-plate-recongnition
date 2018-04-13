# -*- coding: utf-8 -*-

"""
Definition of the neural networks.
定义神经网络

"""

import tensorflow as tf
import common

def print_activation(t):
    print (t.op.name,' ',t.get_shape().as_list())

# 窗口大小
WINDOW_SHAPE = (64, 128)


# 权重变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏置变量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 2d卷积,使用全0添加，输出矩阵尺寸为输入的一半
def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                        padding=padding)

# 最大化池化
def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


# 均值池化
def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


# 模型卷积网
#def convolutional_layers(x, regularizer):
def convolutional_layers(x):

    # 第一层卷积，卷积核尺寸为5*5，输入3，输出为48（第三个参数为当前层的深度，3个RGB通道；第四个参数为滤波器的深度）
    W_conv1 = weight_variable([5, 5, 3,48])
    b_conv1 = bias_variable([48])
    print_activation(x)

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))
    print_activation(h_pool1)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 2), stride=(2, 2))
    print_activation(h_pool2)

    # 第三层卷积
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))
    print_activation(h_pool3)

    # Densely connected layer，全连接的结点为2048（设定的）
    W_fc1 = weight_variable([34 * 9 * 128, 2048])
    #if regularizer != None: tf.add_to_collection('losses', regularizer(W_fc1))
	
    b_fc1 = bias_variable([2048])

    conv_layer_flat = tf.reshape(h_pool3, [-1, 34 * 9 * 128])
	
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)
    print_activation(h_fc1)

    # Output layer
    W_fc2 = weight_variable([2048,  7 * common.num_class])
    #if regularizer != None: tf.add_to_collection('losses', regularizer(W_fc2))
	
    b_fc2 = bias_variable([ 7 * common.num_class])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    print_activation(y)

    return y


