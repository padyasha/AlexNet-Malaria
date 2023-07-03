import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

"""
Configuration settings
"""

# data params
data_size='4_new'

# Learning params
num_classes=2
batch_size = 1

train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2','conv1']
# train_layers = ['fc8', 'fc7','fc6']

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)


# Initialize an saver for store model checkpoints
saver = tf.train.Saver()


if data_size == '256': rgb_mean = 127.5093
elif data_size == '128': rgb_mean = 127.0391
elif data_size == '71': rgb_mean = 127.0901
elif data_size == '71_new': rgb_mean = 124.9396
elif data_size == '1': rgb_mean = 32.1493
elif data_size == '2': rgb_mean = 30.4396
elif data_size == '3': rgb_mean = 12.0638
elif data_size == '1_new': rgb_mean = 32.1265
elif data_size == '2_new': rgb_mean = 30.2056
elif data_size == '3_new': rgb_mean = 11.9783
elif data_size == '4_new': rgb_mean = 26.5960


def imrd(path):
    images = np.ndarray([1, 227, 227, 3])
    img= cv2.imread(path)
    img = cv2.resize(img,(227,227))
    img = img.astype(np.float32)
    img = img-rgb_mean
    images[0] = img
    return images

class_list = 'Healthy'

def saveim(name,data):
    sizes = np.shape(data)   
    fig=plt.figure(figsize=(1,1))
    ax=plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(name, dpi = sizes[0])
    plt.close()

mainpath1='data/data_'+data_size+'/test/'+class_list+'/5252.png'
mainpath2='data/data_'+data_size+'/test/'+class_list+'/5253.png'
mainpath=[mainpath1,mainpath2]

# Start Tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'result/data_'+data_size+'/checkpoint/model_epoch80.ckpt')
    for idx in range(2):
        images=imrd(mainpath[idx])
        conv1 = sess.run(model.conv1, feed_dict={x:images, keep_prob:1.0})
        conv5 = sess.run(model.conv5, feed_dict={x:images, keep_prob:1.0})
        for filter in range(64):
            data1=conv1[0,:,:,filter]
            data2=conv5[0,:,:,2*filter]
            saveim('result/data_'+data_size+'/featuremap/'+class_list+'/conv1_'+mainpath[idx].split('/')[-1].split('.')[0]+'_'+str(filter)+'.png',data1)
            saveim('result/data_'+data_size+'/featuremap/'+class_list+'/conv5_'+mainpath[idx].split('/')[-1].split('.')[0]+'_'+str(2*filter)+'.png',data2)