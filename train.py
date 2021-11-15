import os
import tensorflow as tf
import numpy as np
from PIL import Image
from ops import *
import random

os.environ['CUDA_VISIBLE_DEVICES']='0'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

light_image_dir = "./dataset/high/"
dark_image_dir = "./dataset/low/"
checkpoint_dir = "./log/"
result_dir = "./result/"

# 超参数
batchsize = 30# 批大小
sum_epoch = 500
learning_rate = 0.0001 # 学习率
train_images = 1 # 训练图像的数量
ps = 256 # 图像裁剪的尺寸
cut_images = 100 # 单张图像裁剪的数量
beta1 = 0.5 # 优化器参数

#辅助变量
s = 0 # 计数
count = 0 # 训练计数

# 打乱输入的图像顺序
number_list = [j for j in range(0,train_images)]
random.shuffle(number_list)

light_image_list = []
dark_image_list = []
light_patch_list = []
dark_patch_list = []


for i in range(0,train_images):
    # 归一化
    light_image_list.append(np.array(Image.open(light_image_dir + "%d.png" % i), dtype="float32") / 255.0)
    dark_image_list.append(np.array(Image.open(dark_image_dir + "%d.png" % i), dtype="float32") / 255.0)

# 裁剪
for j in range(cut_images):
    for i in number_list:
        H, W, C = light_image_list[i].shape
        xx = random.randint(0, W - ps)
        yy = random.randint(0, H - ps)
        light_patch_list.append(light_image_list[i][yy:yy+ps,xx:xx+ps,:])
        dark_patch_list.append(dark_image_list[i][yy:yy+ps,xx:xx+ps,:])

# 翻转
for i in range(len(light_patch_list)):
    if np.random.randint(3, size=1)[0] == 0:
        light_patch_list[i] = np.flip(light_patch_list[i], axis=1)
        dark_patch_list[i] = np.flip(dark_patch_list[i], axis=1)
    if np.random.randint(3, size=1)[0] == 1:
        light_patch_list[i] = np.flip(light_patch_list[i], axis=0)
        dark_patch_list[i] = np.flip(dark_patch_list[i], axis=0)

# 打乱顺序
randnum = random.randint(0,100)
random.seed(randnum)
random.shuffle(light_patch_list)
random.seed(randnum)
random.shuffle(dark_patch_list)


batches_count = int(len(light_patch_list) / batchsize)


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

def discriminator(input, is_training=True, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("discriminator", reuse=reuse):
        net = lrelu(bn(conv2d(input, 32, 3, 3, 1, 1, name='d_conv1'), is_training=is_training, scope='d_bn1'))
        net = lrelu(bn(conv2d(net, 32, 3, 3, 1, 1, name='d_conv1_2'),is_training=is_training, scope='d_bn1_2'))
        pool1 = pool_max(net, name='d_pool1')
        net = lrelu(bn(conv2d(pool1, 64, 3, 3, 1, 1, name='d_conv2'), is_training=is_training, scope='d_bn2'))
        net = lrelu(bn(conv2d(net, 64, 3, 3, 1, 1, name='d_conv2_2'), is_training=is_training, scope='d_bn2_2'))
        pool2 = pool_max(net, name='d_pool2')
        net = lrelu(bn(conv2d(pool2, 128, 3, 3, 1, 1, name='d_conv3'), is_training=is_training, scope='d_bn3'))
        net = lrelu(bn(conv2d(net, 128, 3, 3, 1, 1, name='d_conv3_2'), is_training=is_training, scope='d_bn3_2'))
        pool3 = pool_max(net, name='d_pool3')
        net = lrelu(bn(conv2d(pool3, 64, 3, 3, 1, 1, name='d_conv4'), is_training=is_training, scope='d_bn4'))
        net = lrelu(bn(conv2d(net, 64, 3, 3, 1, 1, name='d_conv4_2'), is_training=is_training, scope='d_bn4_2'))
        pool4 = pool_max(net, name='d_pool4')
        net = lrelu(bn(conv2d(pool4, 3, 3, 3, 1, 1, name='d_conv5'), is_training=is_training, scope='d_bn5'))
        net = tf.reshape(net, [batchsize, -1])
        net = lrelu(bn(linear(net, 256, scope='d_fc1'), is_training=is_training, scope='d_bn6'))
        net = lrelu(bn(linear(net, 64, scope='d_fc2'), is_training=is_training, scope='d_bn7'))
        out_logit = linear(net, 1, scope='d_fc3')
        out = tf.nn.sigmoid(out_logit)

        return out, out_logit


input_light_image = tf.placeholder(tf.float32, shape=[batchsize, ps, ps, 3])
input_dark_image = tf.placeholder(tf.float32, shape=[batchsize, ps, ps, 3])

# 网络输出
G = generator(input_dark_image, is_training=True)
D_real, D_real_logits= discriminator(input_light_image,is_training=True)
D_fake, D_fake_logits = discriminator(G, is_training=True)

# 判别网络损失
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
d_loss = d_loss_real + d_loss_fake

# 生成网络损失
g_loss = tf.reduce_mean(tf.square(input_light_image - G)) +  \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))


t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
        .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
        .minimize(g_loss, var_list=g_vars)


with tf.Session(config = tf_config) as sess:
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    for epoch in range(sum_epoch):
        for i in range(0,batches_count):
            _,G_current1,output = sess.run([g_optim,g_loss,G],
                                                feed_dict={ input_dark_image: dark_patch_list[i*batchsize:(i+1)*batchsize] ,
                                                            input_light_image: light_patch_list[i*batchsize:(i+1)*batchsize]})
            _,G_current2 = sess.run([d_optim,d_loss],
                                                feed_dict={ input_dark_image: dark_patch_list[i*batchsize:(i+1)*batchsize] ,
                                                            input_light_image: light_patch_list[i*batchsize:(i+1)*batchsize]})
            count += 1
            if(count % 11) == 0:
                saver.save(sess, checkpoint_dir + 'model.ckpt')
                print('%d %d Loss1=%.3f Loss2=%.3f' % (epoch, i, G_current1,G_current2))
        saver.save(sess, checkpoint_dir + 'model.ckpt')
