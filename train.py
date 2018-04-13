#!_*_coding:UTF-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import numpy as np
import common
import Net
import readtxt


# 输入为一张图片
img = tf.placeholder(tf.float32, [1, 72, 272, 3])
label = tf.placeholder(tf.float32, [1, 476])


def get_loss(y, y_):

    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          logits=tf.reshape(y,
                                                     [-1, common.num_class]),
                                          labels=tf.reshape(y_,
                                                     [-1, common.num_class])) #大小（7,）
    digits_loss = tf.reshape(digits_loss, [-1, 7])     #大小(1, 7)

    digits_loss = tf.reduce_sum(digits_loss, 1)        #大小（1, )
    digits_loss=tf.reduce_sum(digits_loss)

    return digits_loss

#

#可见GPU，第几个GPU可见
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('ckpt_path', './Log',"""存放模型的目录""")

tf.app.flags.DEFINE_string('model_name', 'car_model',"""模型的名称""")

def train():

    logit=Net.convolutional_layers(img)
    # print(logit)
    with tf.name_scope('loss'): #损失
        loss=get_loss(logit,label) # logit 维度（1,476） ，label 维度（1,476）
        tf.summary.scalar('loss',loss)
    # loss = get_loss(logit,label) # logit 维度（1,476） ，label 维度（1,476）

    train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

    correct_logit=tf.argmax(tf.reshape(logit, [-1, 7, common.num_class]), 2)   #大小（1,7）
    correct_label=tf.argmax(tf.reshape(label, [-1, 7, common.num_class]), 2)

    #字符准确率
    with tf.name_scope('accuracy'): #损失
        num_correct=tf.equal(correct_label,correct_logit)   #布尔型
        num_correct=tf.cast(num_correct,tf.float32)         #float32
        accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32))  		
        tf.summary.scalar('accuracy',accuracy)	

    saver = tf.train.Saver(max_to_keep=1)   #保存最近i的模型
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(FLAGS.ckpt_path, sess.graph)

        ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_path)
        step = 0
        if ckpt:
            saver.restore(sess=sess, save_path=ckpt)
            step = int(ckpt[len(os.path.join(FLAGS.ckpt_path, FLAGS.model_name)) + 1:])

        lossall=[]
        accurancyall=[]
        for i in range(step, 4000):

            batch = readtxt.random_data()
            _,losst,car_correct,label_correct,logit_correct,result0=sess.run(
                [train_step,loss,num_correct,correct_label,correct_logit,summary_op],
                       feed_dict={img:batch[0],label:batch[1]})#运行模型
            lossall.append(losst)
            cars_correct=np.sum(car_correct)
            accurancyall.append(cars_correct)
            writer.add_summary(result0,i)
            if i % 10 == 0:
                # result0,losst,car_correct,label_correct,logit_correct=\
                #     sess.run([summary_op,loss,num_correct,correct_label,correct_logit],
                #              feed_dict={img: batch[0], label: batch[1]})
                # writer.add_summary(result0,i) #将日志数据写入文件
                loss_mean=np.mean(lossall)
                print ("*****************************************************")
                print (label_correct)
                print (logit_correct)
                print(car_correct)

                # 获取预测值
                label_char = []
                for j in logit_correct[0]:
                    labelchar = common.show_index[j]
                    label_char.append(labelchar)
                label_chars = "".join(label_char)
                format_str = ("y_: %s,   y: %s")
                print(format_str % (batch[2], label_chars))

                format_str = ("step %d, loss is %s,car numbers correct is %d")
                print(format_str % (i, loss_mean, cars_correct))

                print(accurancyall)
                cars_mean=np.sum(accurancyall)/70
                format_str=("step %d, loss is %s,per 10 car numbers correct mean is %f")
                print(format_str%(i,loss_mean,cars_mean))
                ckptname = os.path.join(FLAGS.ckpt_path, FLAGS.model_name)
                saver.save(sess, ckptname, global_step=i)

                lossall=[]
                accurancyall=[]

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()

