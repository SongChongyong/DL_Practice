# coding:utf-8
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

'''
已知xoy平面上的6个点(1,3),(2,4),(3,7),(4,8),(5,11),(6,14), 求一条直线 y=wx+b,
使得这些点沿y轴方向到该直线的距离的平方和最小。
'''
# 6个点的横坐标、纵坐标
x = tf.constant([1,2,3,4,5,6],tf.float32)
y = tf.constant([3,4,7,8,11,14],tf.float32)

# 初始化直线的斜率w和截距b
w = tf.Variable(1.0,dtype=tf.float32)
b = tf.Variable(1.0,dtype=tf.float32)

# 6个点到直线沿y轴方向距离的平方和
loss = tf.reduce_sum(tf.square(y-(w*x+b)))

# 梯度下降法
opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    MSE = []  # 空列表存放每次迭代后的平均平方误差
    for i in range(500):
        sess.run(opti)
        MSE.append(sess.run(loss))
        # 每隔50次打印直线的斜率和截距
        if i%50==0:
            print((sess.run(w)),(sess.run(b)))
        
    # 画出损失函数的值
    plt.figure(1)
    plt.plot(MSE)
    plt.show()
    
    # 画出6个点及最后计算出的直线
    plt.figure(2)
    x_array, y_array = sess.run([x,y])
    plt.plot(x_array, y_array, 'o')
    xx = np.arange(0,10,0.05)
    yy = sess.run(w)*xx + sess.run(b)
    plt.plot(xx,yy)
    plt.show()
        
'''
1.91 1.2
2.055096 0.75369716
2.1164339 0.4910968
2.1518073 0.3396554
2.1722074 0.2523191
2.1839721 0.20195228
2.1907568 0.17290573
2.1946695 0.1561547
2.1969259 0.14649433
2.1982272 0.14092326
'''


   