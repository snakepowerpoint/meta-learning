import tensorflow as tf


def compute_content_loss(styled_image_encode, c_adain_encode):
    return tf.reduce_mean(tf.squared_difference(styled_image_encode, c_adain_encode))

def compute_style_loss(styled_encode_layers, s_encode_layers, epsilon=1e-6):
    losses = {}
    for layer in styled_encode_layers:
        styled, s = styled_encode_layers[layer], s_encode_layers[layer]

        styled_mean, styled_var = tf.nn.moments(styled, axes=[1, 2])
        s_mean, s_var = tf.nn.moments(s, axes=[1, 2])

        styled_std = tf.sqrt(styled_var + epsilon)
        s_std = tf.sqrt(s_var + epsilon)

        mean_loss = tf.reduce_mean(tf.squared_difference(styled_mean, s_mean))
        std_loss = tf.reduce_mean(tf.squared_difference(styled_std, s_std))

        # normalize w.r.t batch size
        n = tf.cast(tf.shape(styled)[0], dtype=tf.float32)
        mean_loss /= n
        std_loss /= n

        losses[layer] = mean_loss + std_loss

    return tf.reduce_sum(list(losses.values()))
