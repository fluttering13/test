import tensorflow as tf
#Generate the filename queue, and read the gif files contents
filepath = tf.data.Dataset.list_files(str('blue_jay.jpg'))
for i in filepath: #read the file
 filename = i.numpy().decode()
image = tf.io.read_file(filename)
image = tf.io.decode_jpeg(image, channels=3)
#Get gray image
image_tensor = tf.image.rgb_to_grayscale(image) #灰起來呀
image_tensor = tf.cast(image_tensor,tf.float32) #記得轉個格式
#Expand one more dim for batch
image_tensor = tf.expand_dims(image_tensor,0)
maxed_tensor = tf.nn.max_pool(image_tensor,[1,2,2,1],[1,2,2,1],"SAME")
averaged_tensor = tf.nn.avg_pool(image_tensor,
 [1,2,2,1],[1,2,2,1],"SAME")
def normalize_and_encode(img_tensor):
 tensortmp = tf.cast(img_tensor, tf.uint8)
 image_dimensions = tf.shape(img_tensor[0]).numpy()
 tensortmp=tf.reshape(tensortmp,image_dimensions)
 return tf.image.encode_jpeg(tensortmp)
#Names for the max and avg files
fname1 =tf.constant("maxpool.jpeg")
fname2 =tf.constant("avgpool.jpeg")
maxwrite = tf.io.write_file(fname1,
 normalize_and_encode( maxed_tensor))
avgwrite = tf.io.write_file(fname2,
 normalize_and_encode(averaged_tensor))