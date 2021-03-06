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


def img_whitening(images):
    # 图像白化
    adjusted = tf.image.per_image_standardization(images)
    # plt.imshow(adjusted.eval())
    # plt.show()
    return  adjusted

def img_up_down(images ,type):
    if type == "random":
        flipped = tf.image.random_flip_up_down(images)
    else:
        flipped = tf.image.flip_up_down(images)
    # plt.imshow(flipped.eval())
    # plt.show()
    return flipped

def img_left_right(images ,type):
    if type == "random":
        flipped = tf.image.random_flip_left_right(images)
    else:
        flipped = tf.image.flip_left_right(images)
    # plt.imshow(flipped.eval())
    # plt.show()
    return flipped

def transpose_image(images):
    flipped = tf.image.transpose_image(images)
    # plt.imshow(flipped.eval())
    # plt.show()
    return flipped

def img_brightness(images , max_delta):
    imaged = tf.cast(images, tf.float32)
    brightness_image = tf.image.random_brightness(imaged, max_delta= max_delta)  # 随机调整亮度函数
    return  brightness_image

def img_contrast(images , lower, upper):
    imaged = tf.cast(images, tf.float32)
    contrast_image = tf.image.random_contrast(imaged, lower=lower, upper=upper)  # 随机调整对比度函数
    return  contrast_image

# image_raw_data = tf.read_file(img_dir)

def main():
    image_raw_data = tf.gfile.FastGFile(img_dir,'rb').read()
    with tf.Session() as Sess:
        # 解码
        img_data = tf.image.decode_jpeg(image_raw_data)
        # 类型转化
        img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

        adjusted = img_contrast(img_data,0.2,0.8)
        plt.imshow(adjusted.eval())
        plt.show()

if __name__ == '__main__':
  main()