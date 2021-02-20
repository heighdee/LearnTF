'''
# 实现：
#   用 tensorflow 的 Keras 快速构建模型
#
# 模型
#   优化算法：adam优化算法，一种随机梯度下降算法的扩展
#   经验误差：sparse_categorical_crossentrop, 实际值和计算值的交叉熵
#   评估指标: accruacy(准确率)
#
# 数据集：
#   训练集：60000 个样本
#   测试集：10000 个样本
#
# 说明：
#   本实现是 tensorflow 官方教程的第一个示例程序。利用 keras 能快速搭建模型
#   自己重写了一个加载本地数据集的函数
'''
#===========================================================================================

import tensorflow as tf
import numpy as np

def load_data(x_data, y_data):
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
    path = '/Users/jianhaidi/GitHubData/LearnTF/MNIST/Data/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    #1 加载数据
    x_train, y_train = load_data(path + files[0], path + files[1])  # 加载训练集
    x_test, y_test = load_data(path + files[2], path + files[3])  # 加载测试集
    x_train, x_test = x_train / 255.0, x_test / 255.0


    #2 定义模型
    '''
    A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2), #防止过拟合
        tf.keras.layers.Dense(10, activation='softmax') #输出层还是用 softmax 处理
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #3 训练和评估
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)