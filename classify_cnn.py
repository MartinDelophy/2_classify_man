
import tensorflow as tf
import os
import path
import numpy as np
from sklearn.model_selection import train_test_split 


## based in https://blog.csdn.net/lxdssg/article/details/81233837


"""
定义一个遍历文件夹下所有图片，并转化为矩阵，压缩为特定大小，并传入一个总
的矩阵中去的函数
"""      
def creat_x_database(rootdir,resize_row,resize_col):
    #列出文件夹下所有的，目录和文件
    list = os.listdir(rootdir)
    #创建一个随机矩阵，作为多个图片转换为矩阵后传入其中
    database=np.arange(len(list)*resize_row*resize_col*3).reshape(len(list)
    ,resize_row,resize_col,3)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])    #把目录和文件名合成一个路径
        if os.path.isfile(path):                ##判断路径是否为文件
            image_raw_data = tf.gfile.FastGFile(path,'rb').read()#读取图片
            with tf.Session() as sess:
                img_data = tf.image.decode_jpeg(image_raw_data)#图片解码
                #压缩图片矩阵为指定大小
                resized=tf.image.resize_images(img_data,[resize_row,resize_col],method=0)
                database[i]=resized.eval()
    return database                          

def creat_y_database(length,classfication_value,one_hot_value):
    #创建一个适当大小的矩阵来接收
    array=np.arange(length*classfication_value).reshape(length,classfication_value)
    for i in range(0,length):
        array[i]=one_hot_value #这里采用one hot值来区别合格与不合格
    return array


'''
初步打造一个卷积神经网，使其能够对输入图片进行二分类
'''
#计算准确率
 
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result          
#定义各参数变量并初始化            
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 
 
 
#创建训练集
fail_x_data=creat_x_database('./man',128,128)
true_x_data=creat_x_database('./other',128,128) 
x_data=np.vstack((fail_x_data,true_x_data))  #两个矩阵在列上进行合并
#创建标签集    
fail_y_data=creat_y_database(fail_x_data.shape[0],2,[0,1])
true_y_data=creat_y_database(true_x_data.shape[0],2,[1,0])
y_data=np.vstack((fail_y_data,true_y_data))
#划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.1,random_state=0)
#train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
    
    
xs = tf.placeholder(tf.float32, [None, 128,128,3])/255 #归一化
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
#x_image = tf.reshape(xs, [-1, 50, 50, 3])
 
W_conv1 = weight_variable([5,5, 3,32]) # patch 5x5, in size 3, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # output size 128x128x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 64x64x32
 
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 64x64x64
h_pool2 = max_pool_2x2(h_conv2) #32x32x64
 
W_fc1 = weight_variable([32*32*64, 1024])
b_fc1 = bias_variable([1024])
   
h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction+ 1e-10), reduction_indices=[1]))
#由于 prediction 可能为 0， 导致 log 出错，最后结果会出现 NA      
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)    
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
 
for i in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
    if i % 1 == 0:
        print(compute_accuracy(x_test,y_test))
        print(sess.run(cross_entropy, feed_dict={xs: x_train, ys: y_train, keep_prob: 1}))