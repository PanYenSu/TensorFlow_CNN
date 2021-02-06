import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#CIFAR100 小图像分类数据集
#50,000 张 32x32 彩色训练图像数据，以及 10,000 张测试图像数据，总共分为 100 个类别。
#train_images和test_images: uint8 数组表示的 RGB 图像数据，尺寸为 (num_samples, 3, 32, 32) 或 (num_samples, 32, 32, 3)，基于 image_data_format 后端设定的 channels_first 或 channels_last。
#train_labels和test_labels: uint8 数组表示的类别标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1  歸一化像素值在0到1之間
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# 生成一张10*10的图
plt.figure(figsize=(10,10))
for i in range(25):
    #Figure 对象可以包含多个子图(Axes), 可以用 subplot()绘制 subplot(numRows, numCols, plotNum)
    #subplot(行, 列, 编号)  编号：照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    plt.subplot(5,5,i+1)
    #设置刻度值即坐标轴刻度 xticks yticks
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()