import tensorflow as tf
import numpy as np
import DataLoad

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()  # 关闭 eager 模式
    #tf.compat.v1.enable_eager_execution()
    path = '/Users/jianhaidi/GitHubData/LearnTF/MNIST/Data/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    x, y_ = tf.placeholder(tf.float32, shape=[None, 784]), tf.placeholder(tf.float32, shape=[None, 10])

    traindata = DataLoad.load_data(path+files[0], path+files[1]) #加载训练集
    traindataset = tf.data.Dataset.from_tensor_slices(traindata).batch(100)
    iterator = traindataset.make_initializable_iterator()
    features, labels = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        sess.run(iterator.initializer)
        print(sess.run(labels))