import tensorflow as tf
import numpy as np
import tensorflow.keras.utils as utils
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 200
test_size = 400

cnn_layer_size = 2
max_pool_size = 4 # the inner two values of ksize

###########################
######## Functions ########
###########################

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, c1,c2,c3,cfc, p_keep_conv, p_keep_hidden):
    strides = [1,max_pool_size,max_pool_size,1]
    weights = [c1,c2,c3]

    layer = tf.nn.relu(tf.nn.conv2d(X, c1,
                        strides=[1, 1, 1, 1], padding='SAME'))

    layer = tf.nn.max_pool(layer, ksize=strides,
                        strides=strides, padding='SAME')
    layer = tf.nn.dropout(layer, p_keep_conv)

    if(cnn_layer_size > 1):
        for i in range(1,cnn_layer_size):
            layer = tf.nn.relu(tf.nn.conv2d(layer, weights[i],
                                strides=[1, 1, 1, 1], padding='SAME'))
            layer = tf.nn.max_pool(layer, ksize=strides,
                                strides=strides, padding='SAME')
            layer = tf.nn.dropout(layer, p_keep_conv)

    flaten_layer = tf.reshape(layer, [-1,cfc.get_shape().as_list()[0]])        # normalization

    l5 = tf.nn.relu(tf.matmul(flaten_layer,
                        cfc))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    pyx = tf.matmul(l5,init_weights(shape=[625, 10]))

    return pyx

############################
######## Parameters ########
############################

with tf.name_scope("Data") as scope:                    # Training data shape: (50000, 32, 32, 3)
    (Xtr, Ytr), (Xte, Yte) = cifar10.load_data() # Testing data shape: (10000, 32, 32, 3)
    x_train = Xtr; x_test = Xte
    y_train = utils.to_categorical(Ytr)                 # Training label shape: (50000, 10)
    y_test = utils.to_categorical(Yte)                  # Testing label shape: (10000, 10)
    #x_train = Xtr[:5000]; x_test = Xte[:5000]
    #y_train = utils.to_categorical(Ytr[:5000])                 # Training label shape: (50000, 10)
    #y_test = utils.to_categorical(Yte[:5000])                  # Testing label shape: (10000, 10)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

with tf.name_scope("Placeholders") as scope:
    X = tf.placeholder("float", [None, 32, 32, 3])          # input x
    Y = tf.placeholder("float", [None, 10])                 # output y

with tf.name_scope("Weights") as scope:
    c1 = init_weights([3, 3, 3, 64])
    c2 = init_weights([3, 3, 64, 128])
    c3 = init_weights([5, 5, 128, 256])
    a = 64 * (2 ** (cnn_layer_size-1))
    b = 32 / (max_pool_size ** cnn_layer_size)
    cfc = init_weights([int(a*b*b),625])
    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

with tf.name_scope("Model") as scope:
    py_x = model(X, c1,c2,c3,cfc, p_keep_conv, p_keep_hidden)

with tf.name_scope("Functions") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

print(predict_op)

with tf.name_scope("Summaries") as scope:
    tf.summary.histogram("Convolution weights 1", c1)
    tf.summary.histogram("Convolution weights 2", c2)
    tf.summary.histogram("Convolution weights 3", c3)
    tf.summary.histogram("Flatten weights", cfc)
    #tf.summary.scalar("Cost", cost)

    #tf.summary.scalar("train_op", train_op)
    merged_summaries = tf.summary.merge_all()

# ----------------------------- part 6 ---------------------------------------
conv = tf.nn.relu(tf.nn.conv2d(X, c1,
                        strides=[1, 1, 1, 1], padding='SAME'))

test = tf.Session()
test.run(tf.global_variables_initializer())
filteredImage = test.run(conv, feed_dict={X: x_train[0].reshape([1,32,32,3])})

def takeSecond(ele):
    return ele[1]

l = []
for i in range(64):
    temp = (i,np.mean(filteredImage[:,:,:,i]))
    l.append(temp)

l.sort(reverse=True,key=takeSecond)

for i in range(9):
    imgplot = plt.imshow(filteredImage[0,:,:,l[i][0]])
    plt.show(imgplot)
