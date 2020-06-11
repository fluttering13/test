import tensorflow as tf
filepath=tf.data.Dataset.list_files(str('picture.jpg')) #searching the picture.jpg
for i in filepath:
    filepath=i.numpy().decode()

#解析filepath巨巨
image=tf.io.read_file(filename)
#解碼在這裡
image=tf.io.read_jpeg(image,channels=3)
#定義翻翻樂函數
filpimageupdown=tf.image,filp_up_down(image)
filpimageleftright=tf.image,filp_left_right(image)
#使用他
filpimageupdown = \ 
tf.io.encode_jpeg(filpimageupdown)
filpimageleftright = \
tf.io.encode_jpeg(filpimageleftright)
#存檔
fname=tf.cosntant('A.jpeg')
fwirte=tf.io.write_file(fname,flipimageupdown)