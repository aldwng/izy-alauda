import tensorflow as tf
from tensorflow import keras
from keras import layers


class NaiveTransformerEncoder(layers.Layer):

    def __init__(self, embed_dim, dense_dim, multi_head_num=2, activation_func="relu", **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.multi_head_num = multi_head_num
        self.attention_block = layers.MultiHeadAttention(num_heads=multi_head_num, key_dim=embed_dim)
        self.dense_block = keras.Sequential(
            [layers.Dense(dense_dim, activation=activation_func),
             layers.Dense(embed_dim), ])

        self.layer_norm_1 = layers.LayerNormalization()
        self.layer_norm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_out = self.attention_block(inputs,
                                             inputs,
                                             attention_mask=mask
                                             )
        dense_block_in = self.layer_norm_1(inputs + attention_out)
        dense_block_out = self.dense_block(dense_block_in)
        return self.layer_norm_2(dense_block_in + dense_block_out)

    def get_config(self):
        conf = super().get_config()
        conf.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "multi_head_num": self.multi_head_num,
        })

        return conf
