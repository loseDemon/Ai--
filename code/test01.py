import tensorflow as tf

'''
    张量的定义：多维的数组(列表)  阶：张量的维数 (几个括号就是几阶的)
    维数      阶         名字                例子
    0-D       0         标量 scalar          s = 123
    1-D       1         向量 vector          s = [1,2,3]
    2-D       2         矩阵 matrix          s = [[1,2,3],[4,5,6]]
    n-D       n         张量 tensor          s = [[[...]]] (n个)
'''

'''
x.shape 查看的是矩阵和向量的维数：向量的维数是指向量分量的个数，比如 (1,2,3,4)' 是一个4维向量
矩阵的维数是指它的行数与列数,比如:它的维数是 2*3，在数学中,矩阵的维数就是矩阵的秩
                                [ 1 2 3
                                  4 5 6 ]
'''
# 用于创建一个tensor张量
x = tf.constant([[1, 2, 3],
                 [2, 2, 3]])
print(x)

# 常用函数
# tf.reduce_mean 计算张量在指定维度的平均值：axis = 0纵轴，axis = 1 横轴
# tf.reduce_mean(张量，axis = 操作轴)
print("张量X的平均值：", tf.reduce_mean(x))  # 不指定则计算整个张量的平均值
print("张量X在纵轴的平均值：", tf.reduce_mean(x, axis=0))
print("张量X在横轴的平均值：", tf.reduce_mean(x, axis=1))

# 计算张量沿着指定维度的和：tf.reduce_sum(张量名字，axis = 操作轴)
print("张量X在纵轴的平均值：", tf.reduce_sum(x, axis=0))
print("张量X在横轴的平均值：", tf.reduce_sum(x, axis=1))

# tf.Variable(初始值)将变量标记为“可训练”，被标记的信息会在反向传播中记录梯度信息。神经网络训练中，常用该函数标记待训练的数据

# 例如初始化W,mean=平均值，stddev=标准差
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))

# Tensorflow提供的数学函数: 四则运算 + 平方(square)、次方pow(张量,幂次)、开方sqrt()
# 四则运算(只有维度相同的元素才可以做四则运算)： add,subtract,multiply,divide
a = tf.ones([1, 3])  # 1行3列  元素全是1
b = tf.fill([1, 3], 3.0)  # 1行3列  元素全是3.0

print(a)
print(b)

# 张量相加
print(tf.add(a, b))
# 张量相减
print(tf.subtract(a, b))
# 张量相乘
print(tf.multiply(a, b))
# 张量除法
print(tf.divide(a, b))

# 平方、n次方、开方
c = tf.fill([1, 2], 3.)
print(c)
print(tf.square(c))
print(tf.pow(c, 3))
print(tf.sqrt(c))

# 实现两个矩阵相乘 tf.matmul(矩阵1，矩阵2)
matrix01 = tf.ones([3, 2])
matrix02 = tf.fill([2, 3], 3.)

print(tf.matmul(matrix01, matrix02))


w = tf.constant(5, dtype=tf.float32) # 此时定义w为一个一维张量(标量)
print("w = ", w)