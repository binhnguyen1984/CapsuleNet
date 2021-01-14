"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from Utils import squash

class CapsuleLayer(Layer):
    def __init__(self, caps_num, vec_len, routing_rounds = 2, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.caps_num = caps_num
        self.vec_len = vec_len
        self.routing_rounds = routing_rounds
    
    def build(self, input_shape):
        init_sigma = .1
        W_init = tf.random.normal(shape=(1, input_shape[1], self.caps_num, self.vec_len, input_shape[2]),
                             stddev=init_sigma, dtype=tf.float32)
        self.W = tf.Variable(W_init, name='W')
        self.input_caps_num = input_shape[1]
        self.input_caps_len = input_shape[2]
        self.build = True

    def call(self, input_layer, **kwargs):
        return self.routing(input_layer, self.routing_rounds)
    
    def get_capsule_predicted(self, input_layer):
        '''
    
        Parameters
        ----------
        input_layer : matrix of the shape [batch_size, caps_num, caps_dim]
        Returns
        -------
        a capsule layer.
    
        '''
        
        self.batch_size = tf.shape(input_layer)[0]
        W_tiled = tf.tile(self.W, [self.batch_size, 1, 1,1,1], name ='W_tiled')
        caps_output_expanded = tf.expand_dims(input_layer, -1, name='caps_expanded')
        caps_output_tile = tf.expand_dims(caps_output_expanded, 2, name='caps_tile')
        caps_output_tiled = tf.tile(caps_output_tile, [1,1, self.caps_num,1,1], name='caps_tiled')
        return tf.matmul(W_tiled, caps_output_tiled, name='caps_predicted')
        
    def routing(self, input_layer, routing_rounds=2):
        '''
    
        Parameters
        ----------
        input_layer : matrix of the shape [batch_size, caps_num, caps_dim]
        Returns
        -------
        a capsule layer.
    
        '''
        caps_predicted = self.get_capsule_predicted(input_layer)
        raw_weights = tf.zeros([self.batch_size, self.input_caps_num, self.caps_num, 1,1],
                               dtype=tf.float32, name='raw_weights')
        caps_output = None
        for i in range(self.routing_rounds):
            routing_weights = tf.nn.softmax(raw_weights, axis=2, name='routing_weights')
            weighted_predictions = tf.multiply(routing_weights, caps_predicted, name='weighted_predictions')
            weighted_sum = tf.reduce_sum(weighted_predictions, axis = 1, keepdims=True, name='weighted_sum')
            caps_output =  squash(weighted_sum, axis=-2)
            caps_output_tiled = tf.tile(caps_output, [1, self.input_caps_num,1,1,1])
            agreement = tf.matmul(caps_predicted, caps_output_tiled, transpose_a=True, name='agreement')
            raw_weights = tf.add(raw_weights, agreement)
        return caps_output
    
        