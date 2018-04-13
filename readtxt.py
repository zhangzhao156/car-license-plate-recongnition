#coding=utf-8
import random
import numpy as np
import cv2
import common

import tensorflow as tf

f = open('label.txt', 'r')#打开文件
sourceInLines = f.readlines()  # 按行读出文件内容    # 读取所有行，储存在列表中，每个元素是一行。
f.close()#关闭文件

result = []#文件内容列表

for line in sourceInLines:
    data = []
    path, label = line.split()  #path=图片路径 label=DR34FOT
    data.append(path)
    data.append(label)
    result.append(data)


#将一个字符转成one-hot格式
def char_to_vec(c):
    y = np.zeros(common.num_class)
    y[common.index_show[c]] = 1.0
    return y

#****************************************
def code_to_vec(code):
    #code = code.decode('utf-8')
    c2v=[]
    for c in code:
        #c = c.encode('utf-8')
        A=char_to_vec(c)
        c2v=np.hstack((c2v,A))
    c2v = c2v[np.newaxis,:]

    return c2v

#随机获取数据，返回图片读取值和标签
def random_data():

    num = random.randint(0, len(result)-1)

    img_label=result[num] #图片路径和标签

    img_path=img_label[0] #图片路径

    img_data0=cv2.imread(img_path)   #灰度图可以按照彩色图读取三通道，三通道的值相同。
    img_data2=img_data0[np.newaxis,:,:,:]
    img_data=np.cast['float32'](img_data2)
    # img_data1 = cv2.resize(img_data0, (256, 64))
    # img_data2=img_data1[np.newaxis,:,:,:]
    # img_data=np.cast['float32'](img_data2)
    # cv2.imshow("abc",img_data0)
    # cv2.waitKey(0)
    # img_data1 = tf.gfile.FastGFile(img_path0, 'r').read()
    # img_data2 = tf.image.decode_jpeg(img_data1)
    # img_data3 = tf.image.convert_image_dtype(img_data2, dtype=tf.float32)
    # img_data4 = tf.image.resize_images(img_data3, [64, 128])
    # img_data = img_data4[np.newaxis, :, :, :]  # 根据需要扩充维度[1,imgH,imgW,channel]

     #np.cast将数据类型转换成float32
    label0=img_label[1]#标签
    label1=code_to_vec(label0)#热编码
    label2=label1.astype(np.float32)  #将数据类型float64转化为float32

    return img_data,label2,label0  #img_data(1,64,128,3),label2是（1,252）；label0是“MA12JIP”

#
# #
# for i in range(0,5):
#
#     batch = random_data()
#     # format_str = ('img_data=%f label=%f label=%s')
#     # print (format_str % (batch[0], batch[1], batch[2]))
#     print "*********************"
#     # print batch[0].shape
#     # a=batch[0]
#     # b=batch[1]
#     # # print a[0][0]
#     # print b
#     # cv2.imshow('image',batch[3])
#     # cv2.waitKey(0)
#     print (batch[0].shape)
#     print (batch[1].shape)
#     print (batch[2])
