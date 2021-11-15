import os
import tensorflow as tf
import numpy as np
from PIL import Image
from ops import *
import random

os.environ['CUDA_VISIBLE_DEVICES']='0'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

test_image_dir = "./dataset/test/"
checkpoint_dir = "./log/"
result_dir = "./result/"

# 超参数
batchsize = 1 # 批大小
test_images = 1 # 测试图像的数量

test_image_list = []


for i in range(0,test_images):
    # 归一化
    test_image_list.append(np.array(Image.open(test_image_dir + "%d.png" % i), dtype="float32") / 255.0)

def generator(input, is_training=True, reuse=tf.AUTO_REUSE):
    # 4-layers Unet
    with tf.variable_scope("generator", reuse=reuse):
        conv1 = lrelu(bn(conv2d(input, 32, 3, 3, 1, 1, name='g_conv11'), is_training=is_training, scope='g_bn11'))
        conv1 = lrelu(bn(conv2d(conv1, 32, 3, 3, 1, 1, name='g_conv12'), is_training=is_training, scope='g_bn12'))
        pool1 = pool_max(conv1,name = 'g_pool1')

        conv2 = lrelu(bn(conv2d(pool1, 64, 3, 3, 1, 1, name='g_conv21'), is_training=is_training, scope='g_bn21'))
        conv2 = lrelu(bn(conv2d(conv2, 64, 3, 3, 1, 1, name='g_conv22'), is_training=is_training, scope='g_bn22'))
        pool2 = pool_max(conv2,name = 'g_pool2')

        conv3 = lrelu(bn(conv2d(pool2, 128, 3, 3, 1, 1, name='g_conv31'), is_training=is_training, scope='g_bn31'))
        conv3 = lrelu(bn(conv2d(conv3, 128, 3, 3, 1, 1, name='g_conv32'), is_training=is_training, scope='g_bn32'))
        pool3 = pool_max(conv3,name = 'g_pool3')

        conv4 = lrelu(bn(conv2d(pool3, 256, 3, 3, 1, 1, name='g_conv41'), is_training=is_training, scope='g_bn41'))
        conv4 = lrelu(bn(conv2d(conv4, 256, 3, 3, 1, 1, name='g_conv42'), is_training=is_training, scope='g_bn42'))

        up5 = upsample_and_concat(conv4, conv3, 128, 256, name = 'g_up1')
        conv5 = lrelu(bn(conv2d(up5, 128, 3, 3, 1, 1, name='g_conv51'), is_training=is_training, scope='g_bn51'))
        conv5 = lrelu(bn(conv2d(conv5, 128, 3, 3, 1, 1, name='g_conv52'), is_training=is_training, scope='g_bn52'))

        up6 = upsample_and_concat(conv5, conv2, 64, 128, name = 'g_up2')
        conv6 = lrelu(bn(conv2d(up6, 64, 3, 3, 1, 1, name='g_conv61'), is_training=is_training, scope='g_bn61'))
        conv6 = lrelu(bn(conv2d(conv6, 64, 3, 3, 1, 1, name='g_conv62'), is_training=is_training, scope='g_bn62'))

        up7 = upsample_and_concat(conv6, conv1, 32, 64, name = 'g_up3')
        conv7 = lrelu(bn(conv2d(up7, 32, 3, 3, 1, 1, name='g_conv71'), is_training=is_training, scope='g_bn71'))
        conv7 = lrelu(bn(conv2d(conv7, 32, 3, 3, 1, 1, name='g_conv72'), is_training=is_training, scope='g_bn72'))

        out = conv2d(conv7, 3, 1, 1, 1, 1, name='g_conv81')
        return out

input_test_image = tf.placeholder(tf.float32, shape=[batchsize, 400, 600, 3])

# 网络输出
G = generator(input_test_image, is_training=False)

t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if 'g_' in var.name]

with tf.Session(config = tf_config) as sess:
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    for i in range(len(test_image_list)):
        output = sess.run(G,feed_dict={input_test_image:test_image_list[i*batchsize:(i+1)*batchsize]})
        save_images(result_dir + '%d.png' % i, output)
        print("write down %d test image"%i)