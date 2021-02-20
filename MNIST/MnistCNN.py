'''tensorflow 官方教程第一个示例程序
# 实现：
#   tf.keras 可以快速搭建模型。
#   为了加载本地数据集，重写了数据加载部分的函数
#
# 模型：
#   经验误差：交叉熵
#   优化算法：Adam 优化算法，一种随机梯度下降算法的扩展
#   评估指标：准确度
#
# 数据集大小：
#   训练集：60000 个样本
#   测试集：10000 个样本
'''

import tensorflow as tf
import numpy as np

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
    path = '/Users/jianhaidi/GitHubData/LearnTF/MNIST/Data/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    # 1 加载数据
    x_train, y_train = load_data(path + files[0], path + files[1])  # 加载训练集
    x_test, y_test = load_data(path + files[2], path + files[3])  # 加载测试集

    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 2 定义模型
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 3 训练与评估
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)