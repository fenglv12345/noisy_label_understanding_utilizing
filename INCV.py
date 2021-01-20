# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:25:07 2019

@author: 陈鹏飞
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noise_ratio',type=float)
parser.add_argument('--noise_pattern',type=str)
parser.add_argument('--dataset',type=str)
parser.add_argument('--gpu_id',type=int, default=0)
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)

from keras.models import load_model
from my_model import create_model
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import data
import numpy as np
import pandas as pd
import os

""" parameters """
noise_ratio = args.noise_ratio
noise_pattern = args.noise_pattern #'sym' or 'asym'
dataset = args.dataset
batch_size = 128
INCV_epochs = 50
INCV_iter = 4

save_dir = os.path.join('results', dataset, noise_pattern, str(noise_ratio))
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath_INCV = os.path.join(save_dir,'INCV_model.h5')

#################################################################################################################################
""" Data preparation """

if dataset=='cifar10':
    x_train, y_train, _, _, x_test, y_test = data.prepare_cifar10_data(data_dir='data/cifar-10-batches-py')
    Num_top = 1 #using top k prediction
elif dataset=='cifar100':
    x_train, y_train, _, _, x_test, y_test = data.prepare_cifar100_data(data_dir='data/cifar-100-python')
    Num_top = 1 #using top k prediction to select samples
    
y_train_noisy = data.flip_label(y_train, pattern=noise_pattern, ratio=noise_ratio, one_hot=True)
input_shape = list(x_train.shape[1:])
n_classes = y_train.shape[1]
n_train = x_train.shape[0]
#这里就是clean_index在评测的时候会用到，但是在训练的时候不会被用到
clean_index = np.array([(y_train_noisy[i,:]==y_train[i,:]).all() for i in range(n_train)])# For tracking only, unused during training
noisy_index = np.array([not i for i in clean_index])

# Generator for data augmantation
datagen = ImageDataGenerator(width_shift_range=4./32,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=4./32,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=True)  # randomly flip images       

#################################################################################################################################
""" Build model """
# INCV callbacks
class Noisy_acc(Callback):
     
    def on_epoch_end(self, epoch, logs={}):
        if First_half:
            idx = val2_idx # train on the first half while test on the second half 
        else:
            idx = val1_idx
        
        predict = np.argmax(self.model.predict(x_train[idx,:]),axis=1)
        _acc_mix = accuracy_score(np.argmax(y_train_noisy[idx,:],axis=1), predict)
        _acc_clean = accuracy_score(np.argmax(y_train_noisy[idx,:][clean_index[idx],:],axis=1), predict[clean_index[idx]])
        _acc_noisy = accuracy_score(np.argmax(y_train_noisy[idx,:][noisy_index[idx],:],axis=1), predict[noisy_index[idx]])

        print("- acc_mix: %.4f - acc_clean: %.4f - acc_noisy: %.4f\n" % (_acc_mix, _acc_clean, _acc_noisy))
        return
noisy_acc = Noisy_acc()

def INCV_lr_schedule(epoch):
    # Learning Rate Schedule
    lr = 1e-3
    if epoch > 40:
        lr *= 0.1
    elif epoch > 30:
        lr *= 0.25
    elif epoch > 20:
        lr *= 0.5
    print('Learning rate: ', lr)
    return lr
#LearningRateScheduler是学习率规划　是一个回调函数
INCV_lr_callback = LearningRateScheduler(INCV_lr_schedule)

# Define optimizer and compile model
# 优化器 INCV_lr_schedule(0)对应的学习率是0.001
optimizer = optimizers.Adam(lr=INCV_lr_schedule(0), beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#模型
model = create_model(input_shape=input_shape, classes=n_classes, name='INCV_ResNet32', architecture='ResNet32')
#模型 优化器　损失函数　评测标准　正确率
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
#模型初始化
weights_initial = model.get_weights()
# Print model architecture
print('Architecture of INCV-model:')
model.summary()

##################################################################################################################################
""" INCV iteration """
train_idx = np.array([False for i in range(n_train)])
val_idx = np.array([True for i in range(n_train)])
INCV_save_best = True
for iter in range(1,INCV_iter+1):
    
    print('INCV iteration %d including first half and second half. In total %d iterations.'%(iter,INCV_iter))
    #挑选出验证集
    val_idx_int = np.array([i for i in range(n_train) if val_idx[i]]) # integer index
    #打乱
    np.random.shuffle(val_idx_int)
    #分成两份
    n_val_half = int(np.sum(val_idx)/2)
    val1_idx = val_idx_int[:n_val_half] # integer index
    val2_idx = val_idx_int[n_val_half:] # integer index
    #第一半数据集训练
    #Train model on the first half of dataset
    First_half = True
    print('Iteration ' + str(iter) + ' - first half')
    # reset weights
    model.set_weights(weights_initial)
    #对应的训练集为train+vad1 验证集为vad2 训练INCV_epochs轮　训练一轮会有checkpoints best保存，noisy_acc的计算 以及学习率的回调在训练过程中随着epoch的变化会更新
    results = model.fit_generator(datagen.flow(np.concatenate([x_train[train_idx,:],x_train[val1_idx,:]]),
                                                  np.concatenate([y_train_noisy[train_idx,:],y_train_noisy[val1_idx,:]]),
                                                  batch_size = batch_size),
                                  epochs = INCV_epochs,
                                  validation_data=(x_train[val2_idx,:], y_train_noisy[val2_idx,:]),
                                  callbacks=[ModelCheckpoint(filepath=filepath_INCV, monitor='val_acc', verbose=1, save_best_only=INCV_save_best),
                                             noisy_acc,
                                             INCV_lr_callback])
    # Select samples of 'True' prediction
    del model
    model = load_model(filepath_INCV)    
    y_pred = model.predict(x_train[val2_idx,:])
    cross_entropy = np.sum(-y_train_noisy[val2_idx,:]*np.log(y_pred+1e-8),axis=1)
    top_pred = np.argsort(y_pred, axis=1)[:,-Num_top:]
    y_true_noisy = np.argmax(y_train_noisy[val2_idx,:],axis=1)
    top_True = [y_true_noisy[i] in top_pred[i,:] for i in range(len(y_true_noisy))]
　　 #选出预测正确的作为下一轮的训练集
    val2train_idx =  val2_idx[top_True]# integer index
    
    # evaluate noisy ratio and compute discard ratio
    if iter == 1:
        #初始的noisy_ratio　这个其实是使用迭代的方法尝试得到noisy_ratio
        eval_ratio = 0.001
        product = np.sum(top_True)/(n_train/2.)
        #这里使用的其实是sys noisy　type对应的test_accuracy对应的计算公式，这里是把复杂的问题简单化了 两方面 各个类别的noisy_ratio和数量都相等，况且这个可以用解方程的方式求得
        if product<=(1/num_classes):
            eval_ratio=(num_classes-1)/num_classes
        else:
            eval_ratio=((num_classes-1)/num_classes)*(1-m.sqrt((num_classes*product-1)/(num_classes-1)))
        #总体来说 这种评估noisy_ratio的方式还是有些拙劣
        while (1-eval_ratio)*(1-eval_ratio)+eval_ratio*eval_ratio/(n_classes/Num_top-1) > product:
            eval_ratio += 0.001
            if eval_ratio>=1:
                break
        print('noisy ratio evaluation: %.4f\n' % eval_ratio)
        #当eval_ratio小于0.5的时候，对应的discard_ratio小于1
        discard_ratio = min(2, eval_ratio/(1-eval_ratio))       
        discard_idx = val2_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]] # integer index
    
    else:
        #问题来了　如果train和discard有交集怎么办？按照算法的思路是允许存在交集的　这个代码的实施没有问题
        discard_idx = np.concatenate([discard_idx, val2_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]]]) # integer index        
    
    print('%d samples selected\n' % (np.sum(train_idx)+val2train_idx.shape[0]))
    
    #Train model on the second half of dataset
    First_half = False
    print('Iteration ' + str(iter) + ' - second half')
    # reset weights
    model.set_weights(weights_initial)
    results = model.fit_generator(datagen.flow(np.concatenate([x_train[train_idx,:],x_train[val2_idx,:]]), 
                                                  np.concatenate([y_train_noisy[train_idx,:],y_train_noisy[val2_idx,:]]),
                                                  batch_size = batch_size),
                                  epochs = INCV_epochs,
                                  validation_data=(x_train[val1_idx,:], y_train_noisy[val1_idx,:]),
                                  callbacks=[ModelCheckpoint(filepath=filepath_INCV, monitor='val_acc', verbose=1, save_best_only=INCV_save_best),
                                             noisy_acc,
                                             INCV_lr_callback])
    # Select samples of 'True' prediction
    del model
    #加载模型
    model = load_model(filepath_INCV)
    #预测结果
    y_pred = model.predict(x_train[val1_idx,:])
    #计算loss
    cross_entropy = np.sum(-y_train_noisy[val1_idx,:]*np.log(y_pred+1e-8),axis=1)
    #对于预测结果按照class进行排序，选出top1的class
    top_pred = np.argsort(y_pred, axis=1)[:,-Num_top:]
    #ｙ_true对应的标签
    y_true_noisy = np.argmax(y_train_noisy[val1_idx,:],axis=1)
    #选出预测结果和标签相同的数据
    top_True = [y_true_noisy[i] in top_pred[i,:] for i in range(len(y_true_noisy))]    
    #两个验证集选出的数据整合在一起
    val2train_idx =  np.concatenate([val1_idx[top_True],val2train_idx])# integer index
    #loss按照从小到大的顺序进行排序，选择最大的discard_ratio个数目　然后舍弃　也有一个舍弃的集合
    discard_idx = np.concatenate([discard_idx, val1_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]]])
    #训练的index
    train_idx[val2train_idx]=True
    #作为训练的就不能够作为验证集
    val_idx[val2train_idx]=False
    #误差类型　均分的误差
    if noise_pattern == 'sym':
        #被抛弃的数据集也不能够作为验证集
        val_idx[discard_idx]=False
    #对于noisy_ratio的计算　因为有gt的存在，所以可以直接计算出来
    print('%d samples selected with noisy ratio %.4f\n' % (np.sum(train_idx),
                                                           (1-np.sum(clean_index[train_idx])/np.sum(train_idx))))
    #误差类型　非均分误差
    if noise_pattern == 'asym' or eval_ratio > 0.6:
        iter_save_best = 1
    elif eval_ratio > 0.3:
        iter_save_best = 2
    else:
        iter_save_best = 4         
        
    if iter==iter_save_best:
        INCV_save_best = False
        
##################################################################################################################################
""" Save INCV results """
INCV_results = pd.DataFrame({'y':np.argmax(y_train,axis=1),
                             'y_noisy':np.argmax(y_train_noisy,axis=1),
                             'select':train_idx,
                             'candidate':val_idx,
                             'eval_ratio':eval_ratio})

INCV_results.to_csv(
        os.path.join(save_dir,dataset+'_'+noise_pattern+str(noise_ratio)+'_INCV_results.csv'),
        index = False,
        columns = ['y','y_noisy','select','candidate','eval_ratio'])
