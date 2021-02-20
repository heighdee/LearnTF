'''
    来源：TensorFlow官方文档中文版
         极客学院（中文官方文档）:http://wiki.jikexueyuan.com/project/tensorflow-zh/
    说明：为适应TensorFlow 2.4.1版本，部分代码做了必要修改。

    建立第一个 tensorflow 程序，建立初印象
'''

import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()  #关闭 eager 模式

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random.uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
#init = tf.compat.v1.initialize_all_variables()
init = tf.compat.v1.global_variables_initializer()
# 启动图 (graph)
sess = tf.compat.v1.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]