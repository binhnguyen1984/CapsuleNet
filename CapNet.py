from __future__ import division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
from CapsuleLayer import CapsuleLayer
from keras.utils import to_categorical
from Utils import safe_norm

class CapNet(object):
    def __init__(self, n_classes, image_size, n_hiddens):
        self.image_size = image_size
        self.n_hiddens = n_hiddens
        self.n_classes = n_classes
        self.n_hiddens = n_hiddens
        self.image_size = image_size
        self.X = tf.placeholder(shape=[None, image_size, image_size, 1], dtype = tf.float32, name='X')
        self.y = tf.placeholder(shape=[None], dtype = tf.int64, name='y')
        self.mask_with_labels = tf.placeholder_with_default(False, shape=())
        self.checkpoint = 'capnet'
        self.batch_size = 64
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.load_mnist()

        
    def load_mnist(self):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
        return (x_train, y_train), (x_test, y_test)
    
    def next_train_batch(self, iter):
        start = iter*self.batch_size
        end = np.min([start+self.batch_size, len(self.X_train)])
        return self.X_train[start:end], self.y_train[start:end]

    def next_test_batch(self, iter):
        start = iter*self.batch_size
        end = np.min([start+self.batch_size, len(self.X_test)])
        return self.X_test[start:end], self.y_test[start:end]
    
    def create_cap_net(self, caps_dims):
        cap1_nums = 32
        cap1_dims = 8
        conv1_params = {
                "filters": 256, 
                "kernel_size": 9,
                "strides": 1,
                "padding": "valid",
                "activation": tf.nn.relu
        }
        
        conv2_params = {
                "filters": cap1_nums * cap1_dims, 
                "kernel_size": 9,
                "strides": 2,
                "padding": "valid",
                "activation": tf.nn.relu
        }

        conv1 =tf.layers.conv2d(self.X, **conv1_params)
        cap1 = tf.layers.conv2d(conv1, **conv2_params)
        cap1_raw = tf.reshape(cap1, [-1,  cap1_nums * 6 * 6 , cap1_dims],
                       name="caps1_raw")
        return CapsuleLayer(caps_num=self.n_classes, vec_len = caps_dims)(cap1_raw)
        
    def margin_loss(self, caps_output):
        m_plus=0.9
        m_minus = .1
        lambda_ = .5
        T = tf.one_hot(self.y, depth = self.n_classes)
        caps_output_norm = safe_norm(caps_output, axis = -2)
        present_error_raw = tf.square(tf.maximum(0., m_plus - caps_output_norm),
                                       name='present_error_raw')
        present_error = tf.reshape(present_error_raw, shape=(-1, self.n_classes),
                                   name='present_error')
        absent_error_raw = tf.square(tf.maximum(0., caps_output_norm - m_minus),
                                     name='absent_error_raw')
        absent_error = tf.reshape(absent_error_raw, shape=(-1, self.n_classes),
                                  name = 'absent_error')
        L = tf.add(T* present_error, lambda_ * (1. - T) * absent_error)
        return tf.reduce_mean(tf.reduce_sum(L, axis = 1), name = 'margin_loss')

    def reconstruction_loss(self, caps_output, caps_dims):
        y_proba = safe_norm(caps_output, axis=-2, name='y_proba')
        y_proba_argmax = tf.argmax(y_proba, axis=2)
        self.y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name='y_pred')
        reconstruction_targets = tf.cond(self.mask_with_labels,
                                         lambda: self.y,
                                         lambda: self.y_pred)
        reconstruction_mask = tf.one_hot(reconstruction_targets, depth=self.n_classes)
        reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [-1,1, self.n_classes, 1,1])
        caps_output_masked = tf.multiply(caps_output, reconstruction_mask_reshaped)
        decoder_input = tf.reshape(caps_output_masked, [-1, self.n_classes * caps_dims])
        n_output = self.image_size * self.image_size
        with tf.name_scope('decoder'):
            hidden = decoder_input
            for n_hidden in self.n_hiddens:
                hidden = tf.layers.dense(hidden, n_hidden, activation = tf.nn.relu)
            decoder_output = tf.layers.dense(hidden, n_output, activation = tf.nn.relu)
            X_flat = tf.reshape(self.X, [-1, n_output])
            squared_error = tf.square(X_flat - decoder_output)
            return tf.reduce_mean(squared_error)
    
    def loss(self, caps_output, caps_dims):
        alpha = 0.0005
        margin_loss = self.margin_loss(caps_output)
        reconstruction_loss = self.reconstruction_loss(caps_output, caps_dims)
        return tf.add(margin_loss, alpha*reconstruction_loss)
    
    def accuracy(self):
        correct = tf.equal(self.y, self.y_pred)
        return tf.reduce_mean(tf.cast(correct, tf.float32))
    
    def train(self):
        caps_dims = 16
        caps_output = self.create_cap_net(caps_dims)
        self.loss_op = self.loss(caps_output, caps_dims)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(self.loss_op)
        self.accuracy_op = self.accuracy()
        init = tf.global_variables_initializer()       
        n_epochs = 20        
        restore_checkpoint = True
        n_iterations = len(self.X_train) // self.batch_size
        n_validation = len(self.X_test) // self.batch_size
        best_lost_val = np.infty
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            if restore_checkpoint and tf.train.checkpoint_exists(self.checkpoint):
                self.saver.restore(sess, self.checkpoint)
            else: init.run()
            for epoch in range(n_epochs):
                for iter in range(n_iterations):
                    X_batch, y_batch = self.next_train_batch(iter)
                    _, loss = sess.run([training_op, self.loss_op], 
                                       feed_dict={self.X:X_batch,
                                                  self.y:y_batch,
                                                  self.mask_with_labels: True})
                    print("\rIteration:{}/{} ({:.1f}%) loss : {:.5f}".format(iter, n_iterations, iter*100/n_iterations, loss), end="")
                loss_vals = []
                acc_vals = []
                for iter in range(n_validation):
                    X_batch, y_batch = self.next_test_batch(iter)
                    loss_val, acc_val = sess.run([self.loss_op, self.accuracy_op], 
                                                 feed_dict = {self.X: X_batch, self.y:y_batch})
                    loss_vals.append(loss_val)
                    acc_vals.append(acc_val)
                    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iter, n_validation, iter*100/n_validation), end=" " *10)
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                print("\rEpoch: {} val accuracy: {:.4f}% loss: {:.6f}{}".format(epoch, acc_val*100, loss_val, 
                      "(improved)" if loss_val < best_lost_val else ""))
                
                if loss_val < best_lost_val:
                    self.save_path = self.saver.save(sess, self.checkpoint)
                    best_lost_val = loss_val
    def evaluate(self):
        n_iterations = len(self.X_test)//self.batch_size
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint)
            loss_tests = []
            acc_tests = []
            for iter in range(n_iterations):
                X_batch, y_batch = self.next_test_batch(iter)
                loss, acc = sess.run([self.loss_op, self.accuracy_op],
                                     feed_dict={self.X: X_batch,
                                                self.y: y_batch})
                loss_tests.append(loss)
                acc_tests.append(acc)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iter, n_iterations, iter*100/n_iterations), end=" " *10)
            loss_test = np.mean(loss_tests)
            acc_test = np.mean(acc_tests)
            print("Test accuracy: {:.4f}% loss: {:.6f}".format(acc_test*100, loss_test))