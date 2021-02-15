'''用MNIST数据集建立 Softmax 回归模型
# 模型
#   经验误差：交叉熵
#   最小化经验误差求解算法：梯度下降法
#
# 数据集大小：
#   训练集：60000 个样本
#   测试集：10000 个样本
'''
#===========================================================================================
import tensorflow as tf
import numpy as np

def load_data(x_data, y_data):
    """
    Arguments:
      x_data: data file
      y_data: labels file

    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

      **x_train, x_test**: uint8 arrays of grayscale image data with shapes
        (num_samples, 784).

      **y_train, y_test**: uint8 arrays of digit labels (integers in range 0-9)
        with shapes (num_samples,).
    """
    import gzip
    labels_path = y_data
    images_path = x_data
    with gzip.open(labels_path, 'rb') as lbpath:
      labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                             offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
      images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                             offset=16).reshape(len(labels), 784)
    onehot_labels = np.zeros([len(labels), 10])
    for i in range(len(labels)):
        onehot_labels[i][labels[i]] = 1
    return images, onehot_labels

def next_batch(batchsize, x_data, y_data):
    """随机从数据集中抽取 batchsize 个样本进行训练
        Arguments:
            batchsize：要抽取的数据数量大小
            x_data: 特征数据
            y_data: 标签

        Returns:
          x_batch, y_batch
    """
    indexs = np.random.randint(0, len(y_data), batchsize)
    x_batch = np.zeros([batchsize, 784])
    y_batch = np.zeros([batchsize, 10])
    for i in range(batchsize):
        x_batch[i] = x_data[indexs[i]]
        y_batch[i] = y_data[indexs[i]]
    return x_batch, y_batch


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()  # 关闭 eager 模式
    path = '/Users/jianhaidi/GitHubData/LearnTF/MNIST/Data/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    #1 加载数据
    x_train, y_train = load_data(path+files[0], path+files[1]) #加载训练集
    x_test, y_test = load_data(path+files[2], path+files[3])# 加载测试集

    #2 为在图中传输数据集的输入 x 和标签 y 创建占位符
    x = tf.compat.v1.placeholder("float", [None, 784])  # x 占位
    y_ = tf.compat.v1.placeholder("float", [None, 10])  # 实际正确值，y 占位

    #3 创建变量来定义Softmax模型
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W)+b)

    #4 定义经验误差：交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    #5 训练过程就是最小化交叉熵的过程，用梯度下降法，求出模型的 W 和 b，使得训练数据的交叉熵最小
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    #6 预测结果比较
    pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    #7 评估模型：准确率
    accuracy = tf.reduce_mean(tf.cast(pred, 'float'))

    #8 初始化变量
    init_op = tf.compat.v1.global_variables_initializer()  # 定义一个初始化操作：该函数并行的初始化tensorflow 图中的所有变量
    training_epochs = 1000
    batch_size = 5000

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)  # 先运行初始化操作
        for i in range(training_epochs):
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))