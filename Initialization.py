import random #random模块用于生产随机数
import os #查找文件

from keras.utils import np_utils #keras是基于python的深度学习库，np_utils类似于tf.one_hot()
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Convolution1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Nadam
from keras import backend as K
import numpy as np #NumPy是用Python进行科学计算的基础软件包
import sys #sys模块负责程序与python解释器的交互，提供了一系列的函数和变量，用于操控python运行时的环境。
import preprocess
import tensorflow as tf

def printn(string):
    sys.stdout.write(string) #输出string
    sys.stdout.flush() #输出刷新

    
    
def Create_Pairs():
#     UM  = domain_adaptation_task
#     cc  = repetition
#     SpC = sample_per_class
    source_path = r'./data/20HP'
    target_path = r'./data/10HP'
    X_train_source, y_train_source, valid_sourceX, valid_sourceY, test_sourceX, test_sourceY = preprocess.prepro(d_path=source_path,
                                                                                                                                                                                length=1024,
                                                                                                                                                                                number=512,      
                                                                                                                                                                                normal=False,
                                                                                                                                                                                rate=[0.9, 0.05,0.05],
                                                                                                                                                                                enc=False,
                                                                                                                                                                                enc_step=28)
    X_train_source,valid_sourceX,test_sourceX = X_train_source[:,:,np.newaxis], valid_sourceX[:,:,np.newaxis], test_sourceX[:,:,np.newaxis]
    X_train_target, y_train_target, valid_targetX, valid_targetY, test_targetX, test_targetY = preprocess.prepro(d_path=target_path,
                                                                                                                                                                                length=1024,
                                                                                                                                                                                 number=128,     
                                                                                                                                                                                 normal=False,
                                                                                                                                                                                 rate=[0.05, 0.05,0.9],
                                                                                                                                                                                 enc=False,
                                                                                                                                                                                 enc_step=28)
    X_train_target,valid_targetX,test_targetX = X_train_target[:,:,np.newaxis], valid_targetX[:,:,np.newaxis], test_targetX[:,:,np.newaxis]
    np.save('./test/test_targetX.npy', test_targetX)
    np.save('./test/test_targetY.npy',test_targetY)
    

    Training_P=[] #创建一个列表
    Training_N=[]
    # print(y_train_source)
    # print(y_train_target)
    for trs in range(len(y_train_source)):
        for trt in range(len(y_train_target)):
            if np.argmax(y_train_source[trs]) == np.argmax(y_train_target[trt]):
                Training_P.append([trs,trt]) #在列表末尾添加元素，而不影响列表中的其他所有元素
            else:
                Training_N.append([trs,trt])
    random.shuffle(Training_N) #shuffle()方法将序列的所有元素随机排序
    Training = Training_P+Training_N[:3*len(Training_P)] #列表组合
    random.shuffle(Training)

    X1=np.zeros([len(Training),1024,1],dtype='float32')
    X2=np.zeros([len(Training),1024,1],dtype='float32')

    y1=np.zeros([len(Training)]) #默认为浮点型,1维的，len
    y2=np.zeros([len(Training)])
    p=np.zeros([len(Training)])

    for i in range(len(Training)):
        in1,in2=Training[i]
        X1[i,:]=X_train_source[in1,:]
        X2[i,:]=X_train_target[in2,:]

        y1[i]=np.argmax(y_train_source[in1])
        y2[i]=np.argmax(y_train_target[in2])
        if np.argmax(y_train_source[in1]) == np.argmax(y_train_target[in2]):
            p[i]=1

    if not os.path.exists('./pairs'): #os.path.exists(path)，如果path存在，返回True；如果path不存在，返回False。
        os.makedirs('./pairs') #os.makedirs()方法用于递归创建目录

    np.save('./pairs/X1.npy', X1) #写入文件X1=train_source
    np.save('./pairs/X2.npy', X2) #X2=train_target

    np.save('./pairs/y1.npy', y1) #y1=train_source
    np.save('./pairs/y2.npy', y2) #y2=train_target
    np.save('./pairs/p.npy', p)

def Create_Model():
         # 模型参数
        # 输入信号的长度和宽度
        sig_rows, sig_cols = 1024, 1
        # 卷积核个数
        nb_filters = 64
        # 池化大小
        pool_size = 2
        # 卷积核大小
        kernel_size = 16
         # 信号种类个数
        nb_classes = 4
        input_shape = (sig_rows, sig_cols)
        model = Sequential()
        model.add(Convolution1D(nb_filters, kernel_size,
                            padding ='valid',
                            input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Convolution1D(nb_filters, kernel_size)) #添加网络层
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        return model
    
def euclidean_distance(vects): #欧式距离，vects向量
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def training_the_model(model):
    nb_classes=4
#     UM = domain_adaptation_task
#     cc = repetition
#     SpC = sample_per_class
    epoch = 100  # Epoch number，1个epoch等于使用训练集中的全部样本训练一次，通俗的讲epoch的值就是整个数据集被重复几次。
    batch_size = 128 #每次训练在训练集中取batch-size个样本训练
    X_test = np.load('./test/test_targetX.npy')
    y_test = np.load('./test/test_targetY.npy')
    X_test = X_test.reshape(X_test.shape[0], 1024, 1)
    #y_test = np_utils.to_categorical(y_test, nb_classes) #将y_test转换为原始的nb_classes=10个类别
#     print(y_test)
#     print('==============y_test==============')
#     print(y_test.shape)

    X1 = np.load('./pairs/X1.npy')
    X2 = np.load('./pairs/X2.npy')

    X1 = X1.reshape(X1.shape[0], 1024, 1)
    X2 = X2.reshape(X2.shape[0], 1024, 1)
    y1 = np.load('./pairs/y1.npy')
    y2 = np.load('./pairs/y2.npy')
    p = np.load('./pairs/p.npy')
    y1 = np_utils.to_categorical(y1, nb_classes) #转换类别
    y2 = np_utils.to_categorical(y2, nb_classes)
    print('Training the model - Epoch '+str(epoch))
    nn=batch_size
    best_Acc = 0.0
    Acc = 0.0
    for e in range(epoch):
        if e % 10 == 0:
            printn(str(e) + '->')
            print(Acc)
            #len(y2) / nn for 循环范围
        for i in range(31):
            loss = model.train_on_batch([X1[i * nn:(i + 1) * nn, :], X2[i * nn:(i + 1) * nn, :]],
                                        [y1[i * nn:(i + 1) * nn], p[i * nn:(i + 1) * nn]])
            
            loss = model.train_on_batch([X2[i * nn:(i + 1) * nn, :], X1[i * nn:(i + 1) * nn, :]],
                                        [y2[i * nn:(i + 1) * nn], p[i * nn:(i + 1) * nn]])
       
        
        Out = model.predict([X_test,X_test])
        y = []
        Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_test, axis=1)
    # + .0000001
        Acc = (len(Acc_v) - np.count_nonzero(Acc_v)) / len(Acc_v)
        if best_Acc < Acc:
            best_Acc = Acc
    print(str(e))
    return best_Acc #最好的精确度