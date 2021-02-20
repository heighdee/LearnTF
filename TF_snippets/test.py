import tensorflow as tf
tf.compat.v1.disable_eager_execution()  #关闭 eager 模式

input1 = tf.compat.v1.placeholder(tf.experimental.numpy.float32)
input2 = tf.constant([1., 2., 3., 4., 5., 6.])
output = tf.math.log(input2)

with tf.compat.v1.Session() as sess:
    #result1 = sess.run([output1], feed_dict={input1:[7], input2:[2]}) #run执行 output 操作，该操作调用的方法所需要的数据被标记为 feed 数据，feed 数据作为 run 的参数提供
    result = sess.run(output)
    print(result)
    print('end')

