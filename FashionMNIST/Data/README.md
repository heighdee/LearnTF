---
typora-root-url: ./Pictures
typora-copy-images-to: ./Pictures
---

### 数据集下载

官网地址：https://github.com/zalandoresearch/fashion-mnist

百度网盘分享：
    链接: https://pan.baidu.com/s/1waEC5WshqOle_TSQ9WD23Q  密码: oi97

### 数据集说明

该数据集包含 10 个类别的 70,000 个灰度图像。这些图像以低分辨率（28x28 像素）展示了单件衣物。

| name                       | shape           | describe                                  |
| -------------------------- | --------------- | ----------------------------------------- |
| train-images-idx3-ubyte.gz | (60000, 28, 28) | 训练集数据，60000张 28*28 图像            |
| train-labels-idx1-ubyte.gz | (60000,)        | 训练集标签，60000 个图像标签，标签值[0~9] |
| t10k-images-idx3-ubyte.gz  | (10000, 28, 28) | 测试集数据，10000张 28*28 图像            |
| t10k-labels-idx1-ubyte.gz  | (10000,)        | 测试集标签，10000 个图像标签，标签值[0~9] |

类别说明：0（T恤\上衣）、1（裤子）、2（套头衫）、3（连衣裙）、4（外套）、5（凉鞋）、6（衬衫）、7（运动衫）、8（鞋）、9（短靴）

<img src="./Pictures/fashion-mnist-sprite.png" alt="机器学习_数据集" style="zoom:100%;" />