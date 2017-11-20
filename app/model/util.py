import tensorflow as tf

def masked_softmax_loss(labels, logits, mask):
    '''Softmax loss with mask'''
    complete_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=logits)
    normalized_mask = mask / tf.reduce_sum(mask)

    normalized_masked_loss = complete_loss * normalized_mask

    return tf.reduce_sum(normalized_masked_loss)

def masked_accuracy(labels, logits, mask):
    '''Accuracy with mask'''
    complete_prediction = tf.equal(x=tf.argmax(labels, 1),
                                   y=tf.argmax(logits, 1))

    complete_prediction = tf.cast(complete_prediction, tf.float32)

    normalized_mask = mask / tf.reduce_sum(mask)

    normalized_prediction = complete_prediction * normalized_mask

    return tf.reduce_sum(normalized_prediction)
