'''TensorFlow 官方教程--基本分类：对服装图像进行分类
数据集：Fashion MNIST 数据集
    训练集：数据（60000，28，28），标签（60000，）
    测试集：数据（10000，28，28），标签（10000，）

模型：神经网络，输出层用 softmax 处理为概率

实现：
    搭建模型：tf.keras
    画图：matplotlib，查看数据集或者绘制图片来更直观的看预测结果和效果
    与在线版的区别：写了load_data()函数处理本地数据集
'''

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def load_data(x_data, y_data):
    """
    Arguments:
      x_data: data file
      y_data: labels file

    Returns:
      Tuple of Numpy arrays: `(x, y)`.
    """
    import gzip
    labels_path = y_data
    images_path = x_data
    with gzip.open(labels_path, 'rb') as lbpath:
      labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                             offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
      images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                             offset=16).reshape(len(labels), 28, 28)
    return images, labels

if __name__ == '__main__':
    path = '/Users/jianhaidi/GitHubData/LearnTF/FashionMNIST/Data/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 1 加载数据
    x_train, y_train = load_data(path + files[0], path + files[1])  # 加载训练集
    x_test, y_test = load_data(path + files[2], path + files[3])  # 加载测试集

    x_train, x_test = x_train / 255.0, x_test / 255.0 #将像素值[0,255] 处理至 [0,1]

    '''
    # 查看训练集前 25 张图片
    plt.figure(figsize=(10, 10)) #生成一个画布
    for i in range(25):
        plt.subplot(5, 5, i + 1) # 添加子图。子图行列（5，5）的第 i+1 个子图
        plt.xticks([]) #横坐标轴刻度设为空
        plt.yticks([]) #纵坐标轴刻度设为空
        plt.grid(False)  #不要网格线
        plt.imshow(X=x_train[i], cmap=plt.cm.binary) #用 cmap 颜色图谱绘制 X
        plt.xlabel(class_names[y_train[i]]) #设置横坐标轴名称
    plt.show()
    '''

    # 2 定义模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)), #图像像素格式二维(28，28) 展为一维 (784,)
        keras.layers.Dense(128, activation='relu'), #全连接层：128 个神经元
        keras.layers.Dense(10) #Logits 层：输出长度为 10 的数组。
    ])

    model.compile(optimizer='adam', #优化器：模型训练往最小化损失函数的方向进行。
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #损失函数：交叉熵。
                  metrics=['accuracy']) #模型评估指标：准确率

    # 3 训练模型
    model.fit(x_train, y_train, epochs=10)

    # 4 评估模型：模型在测试集上的表现
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # 预测结果
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()]) # 附加一个softmax层，将 logits 转换成更容易理解的概率
    predictions = probability_model.predict(x_test)  #预测结果：每一张图的预测结果都是一个包含10 个数字的数组，分别表示每一类的置信度

    '''
    # 查看预测结果
    print("测试集第一张图的预测结果为：")
    print(predictions[0])
    label = np.argmax(predictions[0]) #返回第一张图预测结果中置信度最大的一类
    print("第一张图预测结果：第%d类，即%s"%(label, class_names[label]))
    print("第一张图实际结果：第%d类，即%s"%(y_test[0], class_names[y_test[0]]))
    '''