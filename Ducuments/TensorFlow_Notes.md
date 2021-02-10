# LearnTF

## TensorFlow 基础

tensorflow 使用 **图（graph）**来表示计算任务，在 **会话（session）**中执行graph，使用 **张量（tensor）**表示数据，通过 **变量（variable**）维护状态，通过 feed 和 fetch 来为操作赋值或者获取数据。

### Tensorflow 的两种模式

在 TF2 中，eager 模式是默认开启的的。如果要用图模式，需要先关闭。

图模式性能更好，在分布式训练方面有优势，eager 模式更为直观、方便调试。

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  #关闭 eager 模式
```

#### 图模式

包含两个步骤：

1. **构建图**：创建好所有需要的资源（变量、占位符、算子...）并搭好整个运行逻辑（用什么数据参与什么操作得到什么结果）。图的节点为操作，边为数据。
2. **运行图**：创建会话（*sess*）来获取图的资源，并根据需要（*fetch*）给定相应的输入（*feed*）运行操作（*sess.run()*）得到计算结果。会话会将图的操作分发到 CPU/GPU等设备上并执行。变量的生存周期由 session 管理。

==TF2 的图模式发生重大调整，变量的管理模式和 eager 一样了，还可以很方便地利用 tf.function 很快的将 eager 代码转换成图模式代码。不再需要 *sess.run()* 和 *占位符* 等。==

#### eager 模式

可理解为命令式编程，可以直接运行直观地看到结果。不像图模式，需要创建会话并运行图后才能知道结果。怎么写 python 程序就怎么写 tensorflow 程序。eager 模式下的变量生存周期的管理类似于 Python对象，没有引用指向的时候就自动销毁。



### 张量的简单理解

tensorflow中的所有数据都通过 ***张量（tensor）***来表示

张量的两个参数：阶数（rank）、形状（shape）。

| 阶数 | 数学角度                 | 实例                                    | 描述                                                      | 形状      | 张量的元素个数 |
| ---- | ------------------------ | --------------------------------------- | --------------------------------------------------------- | --------- | -------------- |
| 0    | 纯数字（只有大小）       | m = 888                                 |                                                           |           |                |
| 1    | 向量（一个基本向量）     | v = [1, 2]                              | 1 层；第一层 2 个元素。                                   | [1]       | 1              |
| 2    | 矩阵（两个基本向量）     | v = [[1, 2], [3, 4]]                    | 2 层；第一层 2 个元素，第二层 2 个元素                    | [2, 2]    | 2x2 = 4        |
| 3    | 立体矩阵（三个基本向量） | v = [[[1], [2]], [[3], [4]], [[5],[6]]] | 3 层；第一层 3 个元素，第二层 2 个元素，第 三 层 1 个元素 | [3, 2, 1] | 3x2x1 = 6      |

理解张量：不谈本质，只从建立对张量的印象以便能快速开始TensorFlow 来说，可以简要地将张量理解为多维数组。0 阶张量为单个数字，1 阶张量为一维数组（将单个数字扩展为一组），2 阶张量为二维数组（将一维数组中的每个元素都扩展为一列数组），3 阶张量为三维数组（继续扩展）。

总之引进张量的目的是为了数字化表示一些具象的数据信息，然后可以用作模型的输入、计算、输出。不断往多维扩展是为了更复杂的表示。



### Tensorflow 程序

支持语言：Python 和 C++/C。python 库提供了大量函数来简化 tensorflow 中图的构建过程，更加容易上手。

我们可以构建一个神经网络的图，然后执行阶段反复执行图中的训练操作（模型的训练需要很多次）。

#### 构建图

tensorflow python 库有一个默认图，对大部分程序已经够用。也可以创建和管理多个图。

构建图必须有一个源op（操作），它的输出被传递给其他的 op 做运算。

#### 启动图

创建一个会话（Session 对象），如果不带任何参数，会默认启动默认图。之后调用 session 对象的 run 方法进行图的计算。会话对象使用结束后需要显性关闭。

可以给 op 指定运算设备，机器的运算设备用字符串标识：

| 字符串   | 描述             |
| -------- | ---------------- |
| “/cpu:0” | 机器的 CPU       |
| "/gpu:0" | 机器的第一个 GPU |
| "/gpu:1" | 机器的第二个 GPU |

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  #关闭 eager 模式

matrix1 = tf.constant([[3, 4]]) #创建第一个常量 op，该 op 作为第一个节点加入到默认图中
matrix2 = tf.constant([[2], [2]])#创建另一个常量 op
product = tf.matmul(matrix1, matrix2)#创建一个矩阵乘法 op，前两个常量 op 作为输入

#如果不用 with 语句，那么需要在创建 sess 会话运算结束后显性关闭 sess.close()
with tf.compat.v1.Session() as sess:#启动默认图
    with tf.device('gpu:1'):#指定计算设备
        result = sess.run(product)
        print(result)
```



### 变量的相关处理

变量维护了图执行过程中的状态信息。通常一个模型的参数会表示成一组变量，例如神经网络的权重可以用存储为某个tensor变量，然后模型训练过程不断更新这个变量的值。

涉及处理变量的两个主要的类：

- tf.Variable : 创建、初始化、赋值等
- tf.compat.v1.train.Saver: 保存和加载

#### 创建和初始化

```python
import tensorflow as tf
tensor1 =  tf.random.normal([784, 200], stddev = 0.35) #TF提供了一系列函数来初始化张量。tensor1的 shape 为[784, 200]
Vname = tf.Variable(tensor1, name = "weight")  #tensor1 的形状自动成为变量 V_name 的形状。shape 固定，但行列数可调整。
```

变量的初始化操作必须在模型的所有其他操作之前完成。

可以先添加一个初始化操作，然后在模型开始之前先运行。

```python
init_op = tf.compat.v1.global_variables_initializer() #定义一个初始化操作：该函数并行的初始化所有变量
with tf.compat.v1.Session() as sess:
  sess.run(init_op) #先运行初始化操作
  '''
  模型的其他操作
  '''
```

还可以用其他变量来初始化一个新的变量，使用其他变量的 *initialized_value()* 方法：

```python
weights = tf.Variable(tf.random.normal([784, 200], stddev = 0.35), name = "weight")
w2 = tf.Variable(weight.initialized_value(), name = "w2")
w_twice = tf.Variable(weight.initialized_value(), name = "w_twice") #方法initialized_value() VS 属性值initial_value
```

#### 保存和加载

```python
'''
	变量的创建和初始化
'''
#创建 Saver 对象来保存变量
saver = tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
  sess.run(init_op) #先运行初始化操作
  path = ".../GitHubData/LearnTF/TF_snippets/"
  save_path = saver.save(sess, path + "xxx.ckpt")# 默认情况下保存所有变量
  saver.restore(sess, path + "model.ckpt"
```

1. 默认情况下：会处理 graph 中的所有变量，每一个变量都以该变量被创建时的名称被保存
2. 可以通过不同参数的设置，仅仅保存和加载部分变量，可以创建多个 Saver 对象来管理同一个 graph 中变量的不同子集：通过给构造函数传入python字典实现，键对应使用的名称，值对应被管理的变量。
3. 如果加载的变量只是部分变量，那么需要在其他的操作之前对剩下的变量进行初始化工作。



### Fetch 和 Feed

图计算中，通常需要获取某个操作的计算结果。Fetch 可以在一次 op 运行中获取多个计算结果。

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  #关闭 eager 模式

input1 = tf.constant([3])
input2 = tf.constant([4])
input3 = tf.constant([5])
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)


with tf.compat.v1.Session() as sess:
    result = sess.run([mul, intermed]) #[]括起，run()中可以运行多个 op
    print(result)
```

Feed机制可以利用占位符来实现用一个 tensor 临时替换一个操作的输出，实现临时插入 tensor到图中执行的效果。使用占位符 tf.placeholder()将一个 op 标记为 Feed 操作，然后 feed 数据作为run()方法的参数传入。方法调用结束，feed 消失。

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  #关闭 eager 模式

input1 = tf.compat.v1.placeholder(tf.experimental.numpy.float32)#标记为‘feed’操作
input2 = tf.compat.v1.placeholder(tf.experimental.numpy.float32)
output = tf.add(input1, input2)

with tf.compat.v1.Session() as sess:
    result = sess.run([output], feed_dict={input1:[7], input2:[2]}) 
    #run执行 output 操作，该操作调用的方法所需要的数据被标记为 feed 数据，feed 数据作为 run 的参数传入
    print(result)
```



## 附录

| 类          | 描述                                                 |
| ----------- | ---------------------------------------------------- |
| tf.constant | Creates a constant tensor from a tensor-like object. |
|             |                                                      |
|             |                                                      |
|             |                                                      |
|             |                                                      |
|             |                                                      |

