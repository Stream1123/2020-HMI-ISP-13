import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import os

#选择硬件设备#
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#选择硬件设备#

#用pandas读取csv格式的训练集和测试集#
emnist_train = pd.read_csv('D:/教材/图像处理/Mycode/train/emnist-csv/emnist-balanced-train.csv', header=None)
emnist_test = pd.read_csv('D:/教材/图像处理/Mycode/train/emnist-csv/emnist-balanced-test.csv', header=None)
#用pandas读取csv格式的训练集和测试集#

#切割训练集#
nd1 = emnist_train.iloc[:,1:] #切出除第一列以外的数据
label_train = emnist_train.iloc[:,0] #切出第一列数据作为标签
print("训练集标签的形状：", label_train.shape,'\n'
      "训练集标签的总数：", label_train.size,'\n'
      "训练集标签的维度：", label_train.ndim,'\n') #用于查看标签信息

label_train = pd.get_dummies(label_train) #将测试集标签进行one-hot编码
print("训练集的形状：",label_train.shape) #查看one-hot处理后标签的形状
assert label_train.shape[0] == label_train.shape[0] #确认训练集和训练集标签第一列是否同形状
label_train = label_train.values #将训练集放入数组存储
x_train = nd1.values #将训练集标签放入数组存储
#切割训练集#

print('这里分割训练集和测试集')

#切割测试集#
nd2 = emnist_test.iloc[:,1:] #切出除第一列以外的数据
label_test = emnist_test.iloc[:,0] #切出第一列数据作为标签
print("测试集标签的形状：",label_test.shape,'\n'
      "测试集标签的总数：",label_test.size,'\n'
      "测试集标签的维度：",label_test.ndim,'\n') #用于查看标签信息

label_test = pd.get_dummies(label_test) #将测试集标签进行one-hot编码
print("测试集的形状：",label_test.shape) #查看one-hot处理后标签的形状
assert label_test.shape[0] == label_test.shape[0] #确认测试集和测试集标签第一列是否同形状
label_test = label_test.values #将测试集放入数组存储
x_test = nd2.values  #将测试集标签放入数组存储
#切割测试集#

# np.save('E:/Project/HandwritingRead/NumpyData/label',label_train)
# np.save('E:/Project/HandwritingRead/NumpyData/sample',x_train_reshape)
# np.save('E:/Project/HandwritingRead/NumpyData/label_test',label_test)
# np.save('E:/Project/HandwritingRead/NumpyData/sample_test',x_test_reshape)

# x_v = np.load('NumpyData/sample.npy')
# y_v = np.load('NumpyData/label.npy')
# x_w = np.load('NumpyData/sample_test.npy')
# y_w = np.load('NumpyData/label_test.npy')

xs = tf.placeholder(tf.float32, [None, 784]) #取出一块？*（28*28）的区域用于存放数据
ys = tf.placeholder(tf.float32, [None, 47]) #取出一块？*47的区域用于存放标签


xs_cut = tf.reshape(xs, [-1, 28, 28, 1]) #将xs形状改变为四维数组，第一维自动计算
# xc_cut = tf.reshape(xc, [-1, 28, 28, 1])
#训练网络
conv1 = tf.layers.conv2d(inputs=xs_cut,
                         filters=32,
                         kernel_size=(3, 3),
                         strides=1,
                         activation=tf.nn.relu,
                         padding='same') #?*28*28*32
pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=(2, 2),
                                strides=2,
                                padding='same') #?*14*14*32
conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=64,
                         kernel_size=(2, 2),
                         strides=1,
                         activation=tf.nn.relu,
                         padding='same') #?*14*14*64
conv3 = tf.layers.conv2d(inputs=conv2,
                         filters=128,
                         kernel_size=(2, 2),
                         strides=1,
                         activation=tf.nn.relu,
                         padding='valid') #?*13*13*128
pool2 = tf.layers.max_pooling2d(inputs=conv3,
                                pool_size=(2,2),
                                strides=2,
                                padding='same') #?*7*7*128
re1 = tf.reshape(pool2,[-1,7*7*128]) #将pool2的数组rehape为[?,7*7*128]
flat1 = tf.layers.dense(inputs=re1,
                        units=1024,
                        activation=tf.nn.relu)
# flat2 = tf.layers.batch_normalization(flat1)
flat3 = tf.layers.dense(inputs=flat1,
                        units=47)
out = tf.nn.softmax(flat3)
# conv4 = tf.layers.conv2d(inputs=pool2,
#                          filters=384,
#                          kernel_size=(3, 3),
#                          strides=1,
#                          activation=tf.nn.relu,
#                          padding='same')
# conv5 = tf.layers.conv2d(inputs=conv4,
#                          filters=256,
#                          kernel_size=(3, 3),
#                          strides=1,
#                          activation=tf.nn.relu,
#                          padding='same')
# pool3 = tf.layers.max_pooling2d(inputs=conv5,
#                                 pool_size=(3, 3),
#                                 strides=2,
#                                 padding='same')
# fc1 = tf.layers.dense(inputs=pool3,
#                       units=4096)
# dp1 = tf.layers.dropout(inputs=fc1,
#                         rate=0.1)
# fc2 = tf.layers.dense(inputs=dp1,
#                       units=4096)
# dp2 = tf.layers.dropout(inputs=fc2,
#                         rate=0.1)
# fc3 = tf.layers.dense(inputs=dp2,
#                       units=1024)
# flat = tf.layers.dense(fc3,47)
# out = tf.nn.softmax(fc3)
#训练网络#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = ys,logits=flat3)) #使用交叉熵设置损失函数
train = tf.train.MomentumOptimizer(1e-3,momentum=0.5).minimize(cross_entropy) #降低损失函数
# dataset = tf.data.Dataset.from_tensor_slices((x_v, y_v))
# print(dataset.output_shapes)
# dataset = dataset.shuffle(1).batch(128).repeat()
# iterator = dataset.make_initializable_iterator()
# data_element = iterator.get_next()
#
# dataset_test = tf.data.Dataset.from_tensor_slices((x_w,y_w))
# print(dataset_test.output_shapes)
# dataset_test = dataset_test.shuffle(1).batch(128).repeat()
# iterator_test = dataset_test.make_one_shot_iterator()
# dataset_test_element = iterator_test.get_next()

# def compute_accuracy(v_xs, v_ys):
#     y_pre = sess.run(flat3, feed_dict={xs: v_xs})
#     correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#     return result

correct = tf.equal(tf.argmax(flat3, 1), tf.argmax(ys, 1))
compute_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.70
with tf.Session(config=sess_config) as sess:
    # #sess.run([tf.global_variables_initializer(), iterator.initializer], feed_dict={xs: x_v, ys: y_v})
    # xs_batch,ys_batch = data_element
    # xs_batch_np = np.array(xs_batch,dtype = tf.int64)
    # ys_batch_np = np.array(ys_batch,dtype = tf.int64)
    # print(xs_batch_np.shape)
    # print(ys_batch_np.shape)
    # xs_batch_reshape = xs_batch_np.reshape([None,28,28])
    # ys_batch_reshape = ys_batch_np.reshape([None,1])
    # print(xs_batch_np.dtype)
    # print(ys_batch_np.dtype)
    # sess.run([tf.global_variables_initializer(), iterator.initializer], feed_dict={xs: xs_batch_np, ys: ys_batch_np})
    # for i in range(1000):
    #         x_w_batch, y_w_batch = sess.run(dataset_test_element)
    #         if i % 50 == 0:
    #             print(compute_accuracy(x_w_batch, y_w_batch))
    sess.run(tf.global_variables_initializer()) #初始化变量
    for e in range(100):
        for i in range(100):
            x_train_b = x_train[i*100:(i+1)*100]  #每次取训练集数据100个
            y_train_b = label_train[i*100:(i+1)*100] #每次取训练集标签100个
            sess.run(train,feed_dict={xs: x_train_b,ys: y_train_b}) #将100个数据和100个标签分别送入xs和ys

            if i%20 == 0:
                accuracy = sess.run(compute_accuracy,feed_dict = {xs:x_test,ys:label_test}) #计算精度
                # accuracy = sess.run(compute_accuracy(x_test,label_test))
                print('当前测试精度：',accuracy,e,i)
    saver = tf.train.Saver()
    save_path=saver.save(sess,'D:/教材/图像处理/Mycode/train/SAVE/model.ckpt')
print('存储路径：',save_path)
