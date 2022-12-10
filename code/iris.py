'''
    利用鸢尾花数据集实现前向传播，反向传播，可视化loss曲线
'''

# 导入需要的模块
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据(因为数据不打乱会影响准确率)
# seed:随机数种子，是一个整数，
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
np.random.seed(116)

# 将打乱的数据分为训练集和测试集  前120行为训练集  后30行为测试集
x_data_train = x_data[:-30]
y_data_train = y_data[:-30]
x_data_test = x_data[-30:]
y_data_test = y_data[-30:]

# 转化x的数据类型，如果数据类型不转化后边会因为数据类型不一致报错
x_data_train = tf.cast(x_data_train, tf.float32)
x_data_test = tf.cast(x_data_test, tf.float32)

# 将输入特征和标签进行打包 from_tensor_slices函数可以将“输入特征和标签”对应一一打包 (把数据集分批次，每个批次batch组数据)
train_db = tf.data.Dataset.from_tensor_slices((x_data_train, y_data_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_data_test, y_data_test)).batch(32)

# 生成神经网络的参数  4个输入特征：故输入层为4个节点; 因为3分类，故：输出层有3个神经元
# 用tf.Variable标记参数可训练 (实际上不指定 seed)
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后边画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step,loss_all记录四个step生成的4个loss的和

# 训练部分
for epoch in range(epoch):
    for step, (x_data_train, y_data_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_data_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_data_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 -lr* w1_grad  b = b-lr*b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数w2自更新

    # 每个epoch打印loss信息
    print("Epoch: {},loss: {}".format(epoch, loss_all / 4))
    # 将4个step的loss求平均数   然后记录在此变量中
    train_loss_results.append(loss_all / 4)
    # 将loss设置为0 为下一次epoch做好准备
    loss_all = 0

    # 测试部分
    total_correct = 0  # 测试对的样本个数
    total_number = 0  # 测试对的总样本数
    for x_data_test, y_data_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_data_test, w1) + b1
        y = tf.nn.softmax(y)  # 将前向传播的预测结果转化为概率分布
        pred = tf.argmax(y, axis=1)  # 返回y中最大的索引，即预测的分类
        # 将pred转化为y_test类型的数据
        pred = tf.cast(pred, dtype=y_data_test.dtype)
        # 若是分类正确，则correct=1,否则为0，将bool类型的结果转化为int类型
        correct = tf.cast(tf.equal(pred, y_data_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有的batch的corrct加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_data_test的行数
        total_number = x_data_test.shape[0]

    # 总的准确率为： total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", test_acc)
    print("-----------------------------------")

# 绘制loss曲线
plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

# 绘制Accuracy曲线
plt.title("Acc Curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()
