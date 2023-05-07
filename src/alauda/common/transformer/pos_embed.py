import tensorflow as tf
from keras import layers


class PosEmbedding(layers.Layer):

    def __init__(self, seq_len, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.term_embed = layers.Embedding(input_dim=in_dim, output_dim=out_dim)
        self.pos_embed = layers.Embedding(input_dim=seq_len, output_dim=out_dim)
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.out_dim = out_dim

    def call(self, inputs):
        len = tf.shape(inputs)[-1]
        pos = tf.range(start=0, limit=len, delta=1)
        embed_terms = self.term_embed(inputs)
        embed_pos = self.pos_embed(pos)
        return embed_terms + embed_pos

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        conf = super().get_config()
        conf.update({
            "input_dim": self.in_dim,
            "output_dim": self.out_dim,
            "sequence_length": self.seq_len,
        })

        return conf
