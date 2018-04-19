import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_dir = r'E:\DeepLearning\machine_learning_realize\Tensorflow\resource\image\000001.jpg'

def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image,max_delta=32./255.)#亮度
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)#饱和度
        image = tf.image.random_hue(image,max_delta=0.2)#色相
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)#对比度
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
    return tf.clip_by_value(image,0.0,1.0) #将张量值剪切到指定的最小值和最大值

def preprocess_for_train(image,height,width,bbox):
    #如果没有提供标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])

    #转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    #随机截取图像，减少需要关注的物体大小对图像识别的影响
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),
                                                                    bounding_boxes=bbox)
    distort_image = tf.slice(image,bbox_begin,bbox_size)

    #将随机截图的图像调整为神经网络输入层的大小。大小调整的算法是随机的
    distort_image = tf.image.resize_images(
        distort_image,[height,width],method=np.random.randint(4)
    )
    #随机左右翻转图像
    distort_image = tf.image.random_flip_left_right(distort_image)
    #使用一种随机的顺序调整图像色彩
    distort_image = distort_color(distort_image,np.random.randint(1))
    return distort_image

def image_crop(images, shape):
    # 图像切割
    new_images = []
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        left = np.random.randint(old_image.shape[0] - shape[0] + 1)
        top = np.random.randint(old_image.shape[1] - shape[1] + 1)
        new_image = old_image[left: left + shape[0], top: top + shape[1], :]
        new_images.append(new_image)

    return np.array(new_images)

def image_crop_test(images, shape):
    # 图像切割
    new_images = []
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        left = int((old_image.shape[0] - shape[0]) / 2)
        top = int((old_image.shape[1] - shape[1]) / 2)
        new_image = old_image[left: left + shape[0], top: top + shape[1], :]
        new_images.append(new_image)
    return np.array(new_images)

def image_flip(images):
    # 图像翻转
    for i in range(images.shape[0]):
         old_image = images[i, :, :, :]
         if np.random.random() < 0.5:
              new_image = cv2.flip(old_image, 1)
         else:
             new_image = old_image
         images[i, :, :, :] = new_image
    return images

def image_whitening(images):
    # 图像白化
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        new_image = (old_image - np.mean(old_image)) / np.std(old_image)
        images[i, :, :, :] = new_image
    return images

def image_noise(images, mean=0, std=0.01):
    # 图像噪声
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        new_image = old_image
        for i in range(images.shape[0]):
             for j in range(images.shape[1]):
                for k in range(images.shape[2]):
                     new_image[i, j, k] += random.gauss(mean, std)
        images[i, :, :, :] = new_image
    return images


# image_raw_data = tf.read_file(img_dir)
image_raw_data = tf.gfile.FastGFile(img_dir,'rb').read()

with tf.Session() as Sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())
    plt.imshow(img_data.eval())
    plt.show()
    # 类型转化
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    crop_shape = (24, 24, 3)
    plt.imshow(img_data.eval())
    plt.show()
    adjusted = tf.image.per_image_standardization(img_data)
    plt.imshow(adjusted.eval())
    plt.show()
    # 图像切割
    # images = image_crop(img_data, shape=crop_shape)
    # plt.imshow(images.eval())
    # plt.show()
    # 图像翻转
    # images = image_flip(ima_data)
    # plt.imshow(images.eval())
    # plt.show()
    # # 图像白化
    # images = image_whitening(ima_data)
    # plt.imshow(images.eval())
    # plt.show()
    # # 图像噪声:
    # noise_mean = 0
    # noise_std = 0.01
    # images = image_noise(images, mean=noise_mean, std=noise_std)
    # plt.imshow(images.eval())
    # plt.show()
    # boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    #
    # #运行6次获得6中不同的图像，在图中显示效果
    # for i in range(6):
    #     #将图像的尺寸调整为299*299
    #     result = preprocess_for_train(ima_data,299,299,boxes)
    #
    #     plt.imshow(result.eval())
    #     plt.show()