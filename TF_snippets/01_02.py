'''
    来源：TensorFlow官方文档中文版
         极客学院（中文官方文档）:http://wiki.jikexueyuan.com/project/tensorflow-zh/
    说明：为适应TensorFlow 2.4.1版本，部分代码做了必要修改。

    内容：tensorflow 变量的创建、初始化、保存和加载。
'''

import tensorflow as tf
tf.compat.v1.disable_eager_execution()  #关闭 eager 模式

# 创建和初始化变量
tensor1 = tf.random.normal([2, 3, 4], stddev = 0.35)
weights = tf.Variable(tensor1, name = "weights")

# 用已有变量初始化变量
#weights.initial_value可替换为weights.initialized_value()
w_twice = tf.Variable(weights.initial_value * 0.2, name = "w_twice")

# 保存和加载变量
saver = tf.compat.v1.train.Saver()
init_op = tf.compat.v1.global_variables_initializer() #定义一个初始化操作：该函数并行的初始化所有变量
with tf.compat.v1.Session() as sess:
  sess.run(init_op) #先运行初始化操作
  path = "/Users/jianhaidi/GitHubData/LearnTF/TF_snippets/"
  save_path = saver.save(sess, path + "model.ckpt")
  saver.restore(sess, path + "model.ckpt")
