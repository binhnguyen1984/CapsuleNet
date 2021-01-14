import tensorflow as tf

def safe_norm(s, axis =-1, epsilon = 1e-7, name='safe_norm'):
    with tf.name_scope(name):
        squared_norm = tf.reduce_sum(tf.square(s), axis = axis, keepdims=True)
        return tf.sqrt(squared_norm + epsilon)
    
def squash(capsule, axis=-1, name='squash'):
    with tf.name_scope(name):
        squared_norm = safe_norm(capsule)
        squash_factor = squared_norm/(1.+ squared_norm)
        unit_vector = capsule/squared_norm
        return squash_factor*unit_vector
