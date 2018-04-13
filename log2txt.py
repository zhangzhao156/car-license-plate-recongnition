# -*- coding: utf-8 -*-
import os
import glob

def data_process_filenames(path):
    img_path = path
    # 图片所在的文件夹
    img_train_path= os.path.join(img_path, 'train')

    #获取训练数据的标签，视差图

    img_filenames = sorted(glob.glob(img_train_path + "/*"))#获取path/img/*的所有文件，返回值 list

    #打开label.txt文件
    f = open('label.txt', 'w')

    for i in range(len(img_filenames)):#list的长度

        str_path=img_filenames[i] #第i个文件的路径+名字
        # 符串 例如：'...../img/00000000_GO12ODQ_0.png'

        str_filenames=os.path.splitext(str_path)[0]  #分割路径，返回路径名和文件扩展名的元组
        # 路径名=....../img/00000000_GO12ODQ_0  文件扩展名=.png

        str_file=str_filenames.split('/')#将 路径字符串  分割  分隔符为 '/’ 返回值为list
        filename_str=str_file[-1] #车牌号    “00000000_GO12ODQ_0”
        filename_str=filename_str.split('_')[1]#分割字符串，取车牌“GO12ODQ”
        f.write(str_path)#将图片的路径，写入文件
        f.write(' ')#添加一个空格
        f.write(filename_str) #车牌号[标签]写入文件
        f.write('\n')#转行字符写入文件    换行

    f.close()

#文件读写操作
# f=open('f.txt','w')    # r只读，w可写，a追加
# for i in range(0,10):f.write(str(i)+'\n')
# f.close()

#data_path='/mnt/c/car/Car_48_5layers_Chinese/'
data_path=''

data_process_filenames(data_path)