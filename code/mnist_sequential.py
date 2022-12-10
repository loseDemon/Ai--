# 采用六步法搭建手写数字识别的网络
# 引入需要的包
import tensorflow as tf

# 加载需要的数据集mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 将数据转化为0-1之间的数，归一化处理，神经网络更好的吸收
x_train, x_test = x_train/255.0, x_test/255.0

# 使用Sequential搭建网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 在complie()中配置
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 符合正太分布所以是Flase
    metrics=['sparse_categorical_crossentropy']
)

# 在fit中训练网络
model.fit(x_train, y_train, batch_size=32, epochs=5,validation_data=(x_test, y_test), validation_freq=1)

# 最后打印网络的运行结果
model.summary()
