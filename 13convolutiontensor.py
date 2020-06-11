import tensorflow as tf
#Generate the filename queue, and read the gif files
filepath = tf.data.Dataset.list_files(str('blue_jay.jpg'))
#input the pixels into the program
for i in filepath:
 filename = i.numpy().decode()
image = tf.io.read_file(filename)
image = tf.io.decode_jpeg(image, channels=3)
#Define the kernel parameters
##2Dfn
boudary_enhance=tf.constant([[[[0.1]],[[0.1]]],
 [[[-0.1]],[[-0.1]]]])#horizontal enchance
boudary_enhancetwo=tf.constant([[[[0.1]],[[-0.1]]],[[[0.1]],[[-0.1]]]])
##Edge detection
kernel=tf.constant([[[[-1.]],[[-1.]],[[-1.]]],
 [[[-1.]],[[ 8.]],[[-1.]]],
[[[-1.]],[[-1.]],[[-1.]]]])
edgefn=tf.constant([[[[1]],[[0]],[[-1.]]],
 [[[0]],[[0]],[[0]]],
[[[-1]],[[0]],[[1]]]])
sharpe=tf.constant([[[[0]],[[-1]],[[0]]],
 [[[-1]],[[5.]],[[-1]]],
[[[0]],[[-1]],[[0]]]])

#vertical enchance
#Get first image
image_tensor = tf.image.rgb_to_grayscale(image) #grayscale
image_tensor = tf.cast(image_tensor, tf.float32)
#Expand one more dim for batch
image_tensor = tf.expand_dims(image_tensor,0) #input must be 4 D
#apply convolution, preserving the image size
image_convoluted_tensor=tf.nn.conv2d(image_tensor,sharpe, [1,1,1,1], "SAME")
#Prepare to save the convolution option
fname = tf.constant('blur2.jpeg')
# Cast to uint8 (0.255), previous scalation, because the
# convolution could alter the scale of the final image
tensortmp = \
tf.reduce_max(image_convoluted_tensor).numpy()
tensortmp=255.0*image_convoluted_tensor / tensortmp #normalzation of tensortmo
tensortmp = tf.cast(tensortmp, tf.uint8)
tensortmp = tf.reshape(tensortmp,
 tf.shape(image_tensor[0]).numpy())
out=tf.image.encode_jpeg(tensortmp)
fwrite = tf.io.write_file(fname, out) 