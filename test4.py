import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_images(data_home='images', categories=0):
    image_folders = sorted(os.listdir(data_home))
    total_categories = len(image_folders) if categories == 0 else categories
    images, labels = [], []
    for i in range(0, total_categories):
        category_label = image_folders[i].split('.')[0]
        category_folder = os.path.join(data_home, image_folders[i])
        one_hot_label = np.zeros(total_categories)
        one_hot_label[int(category_label) - 1] = 1
        for image in os.listdir(category_folder):
            # img = cv2.imread(os.path.join(category_folder, image), cv2.IMREAD_COLOR)
            # img = process_image(img)
            # img = cv2.resize(img, image_shape)
            img = cv2.resize(cv2.imread(os.path.join(category_folder, image), cv2.IMREAD_COLOR),image_shape)
            dst = np.zeros((image_shape[0],image_shape[1],3))
            images.append(cv2.normalize(img,dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
            # images.append(img)
            labels.append(one_hot_label)
    return np.array(images), np.array(labels)


def conv_layer(layer_name,filter_shape, filter_num, input_features):
    with tf.name_scope(layer_name):

        with tf.name_scope("cv_weights"):
            w = tf.Variable(tf.truncated_normal([filter_shape[0], filter_shape[1], filter_shape[2],
                                                   filter_num], stddev=init_weight_std))
            w_mean = tf.reduce_mean(w)
            tf.summary.scalar("weight_mean",w_mean)
            tf.summary.histogram("weight histogram", w)

        with tf.name_scope("cv_biases"):
            bias = tf.Variable(tf.constant(init_bias, shape=[filter_num]))
            bias_mean = tf.reduce_mean(bias)
            tf.summary.scalar("bias_mean", bias_mean)
            tf.summary.histogram("bias histogram", bias)

        with tf.name_scope("cv_filters"):
            conv = tf.nn.relu(tf.nn.conv2d(input_features, w, strides=[1, 1, 1, 1], padding='SAME') + bias)
    return conv

def pool_layer(conv):
    with tf.name_scope("pool_layer"):
        h_pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        mean = tf.reduce_mean(h_pool)
        tf.summary.scalar('pool_mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(h_pool - mean)))
        tf.summary.scalar('pool_std', stddev)
        tf.summary.scalar('pool_max', tf.reduce_max(h_pool))
        tf.summary.scalar('pool_min', tf.reduce_min(h_pool))
        tf.summary.histogram('pool_histogram', h_pool)
    return h_pool


def fc_layer(pool_width, pool_height, pool_num ,filter_num, pool):
    with tf.name_scope("fully_connected_layer"):
        with tf.name_scope("fc_weight"):
            w_fc = tf.Variable(tf.truncated_normal([pool_width*pool_height*pool_num,filter_num],stddev=init_weight_std))
            w_mean = tf.reduce_mean(w_fc)
            tf.summary.scalar("weight_mean", w_mean)
            tf.summary.histogram("weight_histogram", w_fc)
        with tf.name_scope('fc_bias'):
            bias_fc = tf.Variable(tf.constant(init_bias, shape=[filter_num]))
            bias_mean = tf.reduce_mean(bias_fc)
            tf.summary.scalar("bias_mean", bias_mean)
            tf.summary.histogram("bias_histogram", bias_fc)
        with tf.name_scope('fc_activation'):
            h_pool_flat = tf.reshape(pool,[-1,pool_width*pool_height*pool_num])
            Wx_plus_b = tf.matmul(h_pool_flat, w_fc) + bias_fc
            tf.summary.histogram('Wx_plus_b', Wx_plus_b)
            h_fc = tf.nn.relu(Wx_plus_b)
            tf.summary.histogram("relu_activation", h_fc)
    return h_fc


def softmax_layer(fc_nums,fc, kp):
    with tf.name_scope('softmax_layer'):
        with tf.name_scope("dropout"):
            tf.summary.scalar('dropout_keep_probability', kp)
            h_fc1_drop = tf.nn.dropout(fc, kp)
            # tf.summary.scalar('fc_drop', h_fc1_drop)
        with tf.name_scope('softmax_weight'):
            w_fc = tf.Variable(tf.truncated_normal([fc_nums,train_categories],stddev=init_weight_std))
            w_mean = tf.reduce_mean(w_fc)
            tf.summary.scalar("weight_mean", w_mean)
            tf.summary.histogram("weight_histogram", w_fc)
        with tf.name_scope('softmax_bias'):
            bias_fc = tf.Variable(tf.constant(init_bias, shape=[train_categories]))
            bias_mean = tf.reduce_mean(bias_fc)
            tf.summary.scalar("bias_mean", bias_mean)
            tf.summary.histogram("bias_histogram", bias_fc)
        with tf.name_scope("softmax_activation"):
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc) + bias_fc)
            tf.summary.histogram('activation', y_conv)
    return y_conv


def draw_tensors(X,Y, kp):
    #first hidden layer
    h1_filters = 64
    h_conv1 = conv_layer("convolutional_layer_1", (5,5,3),h1_filters, X)

    #pooling
    h_pool1 = pool_layer(h_conv1)
    #second hidden layer
    h2_filters = 128
    h_conv2 = conv_layer("convolutional_layer_2", (5,5,h1_filters),h2_filters, h_pool1)
    #pooling
    h_pool = pool_layer(h_conv2)
    #fully connected network
    connected_neurons = 1096
    pool_width = int(image_shape[0]/4)
    pool_height = int(image_shape[1]/4)
    h_fc1 = fc_layer(pool_width,pool_height,h2_filters,connected_neurons,h_pool)
    #sofmax layer
    y_conv = softmax_layer(connected_neurons, h_fc1, kp)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)

    return cross_entropy, y_conv


def train():
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], 3], name='x-input')
        Y = tf.placeholder(tf.float32, shape=[None, train_categories], name='y-input')
        keep_prob = tf.placeholder(tf.float32,name='keep-prob')
        tf.summary.image('input image', X, 10)
    cross_entropy,y_conv = draw_tensors(X,Y,keep_prob)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.7)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    perm = np.arange(total_samples)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('ts_board/train_summary', sess.graph)
        test_writer = tf.summary.FileWriter('ts_board/test_summary')
        sess.run(tf.global_variables_initializer())

        for i in range(500):
            np.random.shuffle(perm)
            train_samples = X_train[perm]
            train_labels = y_train[perm]
            # train_writer.add_summary(summary, i)
            if i % 10 == 0:
                test_summary,acc,log_loss = sess.run([merged, accuracy,cross_entropy],
                                                feed_dict={X: X_test, Y: y_test, keep_prob: 1.0})
                test_writer.add_summary(test_summary,i)
                print('Accuracy at step %s: %s' % (i, acc))
                print("log loss : %f" % (log_loss))
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_summary,_ = sess.run([merged, train_step], feed_dict={X: train_samples, Y: train_labels, keep_prob: 0.5},
                options=run_options, run_metadata=run_metadata)
                lr = sess.run(learning_rate)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(train_summary, i)
                print("learning rate %f" % (lr))
            else:

                train_summary, _ = sess.run([merged, train_step],
                                            feed_dict={X: train_samples, Y: train_labels, keep_prob: 0.5})
                train_writer.add_summary(train_summary,i)

        train_writer.close()
        test_writer.close()




if __name__=="__main__":
    init_bias = 0.01
    init_weight_std = 0.001
    image_shape = (32,32)
    train_categories = 2
    images, labels = load_images(categories=train_categories)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.15, random_state = 42)
    total_samples = len(X_train)
    if tf.gfile.Exists("ts_board"):
        tf.gfile.DeleteRecursively("ts_board")
    train()