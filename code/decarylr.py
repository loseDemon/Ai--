import numpy as np
import tensorflow as tf
print("============学习率指数迭代优化==============")
# 目标 ： 不断优化参数w和b(此处定义b = 1),找到最小的 lr
epoch = 40  # 循环迭代次数
LR_BASE = 0.2
LR_DECRY = 0.99
LR_STEP = 1
w = tf.Variable(tf.constant(5, dtype=tf.float32)) # 此时定义w为一个一维张量(标量)
print("w = ", w)

for epoch in range(epoch):
    lr = LR_BASE * LR_DECRY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)  # 对每一个元素求平方
    grads = tape.gradient(loss, w)

    w.assign_sub(lr * grads)
    print("After %s epoch , w is %f,loss is %f,lr is %f " % (epoch, w.numpy(), loss, lr))

