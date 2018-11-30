import tensorflow as tf
import numpy as np
import tensorflow.keras.utils as utils
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import cifar10
tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 128
test_size = 256

###########################
######## Functions ########
###########################

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)


    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

############################
######## Parameters ########
############################

with tf.name_scope("Data") as scope:                    # Training data shape: (50000, 32, 32, 3)
    (x_train, Ytr), (x_test, Yte) = cifar10.load_data() # Testing data shape: (10000, 32, 32, 3)
    y_train = utils.to_categorical(Ytr)                 # Training label shape: (50000, 10)
    y_test = utils.to_categorical(Yte)                  # Testing label shape: (10000, 10)

with tf.name_scope("Placeholders") as scope:
    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10])

with tf.name_scope("Weights") as scope:
    w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
    w_fc = init_weights([32 * 14 * 14, 625]) # FC 32 * 14 * 14 inputs, 625 outputs
    w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

with tf.name_scope("Model") as scope:
    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden)

with tf.name_scope("Functions") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

#########################
######## Runtime ########
#########################
'''
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(10):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))'''
