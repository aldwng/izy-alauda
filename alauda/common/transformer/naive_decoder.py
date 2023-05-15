import tensorflow as tf
from tensorflow import keras
from keras import layers


class NaiveTransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, multi_head_num=2, activation_func="relu", **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.multi_head_num = multi_head_num
        self.attention_block_1 = layers.MultiHeadAttention(multi_head_num, key_dim=embed_dim)
        self.attention_block_2 = layers.MultiHeadAttention(multi_head_num, key_dim=embed_dim)
        self.dense_block = keras.Sequential(
            [layers.Dense(dense_dim, activation=activation_func),
             layers.Dense(embed_dim), ])

        self.layer_norm_1 = layers.LayerNormalization()
        self.layer_norm_2 = layers.LayerNormalization()
        self.layer_norm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_seq_attention_mask(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        seq_len = shape[1]
        m = tf.range(seq_len)[:, tf.newaxis]
        n = tf.range(seq_len)
        mask = tf.cast(m >= n, dtype="int32")
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        multiples = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, multiples)

    def call(self, inputs, encoder_out, mask=None):
        seq_mask = self.get_seq_attention_mask(inputs)
        if mask is not None:
            padd_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padd_mask = tf.minimum(padd_mask, seq_mask)

        attention_out_1 = self.attention_block_1(query=inputs,
                                                 value=inputs,
                                                 key=inputs,
                                                 attention_mask=seq_mask,
                                                 )
        attention_out_1 = self.layer_norm_1(inputs + attention_out_1)

        attention_out_2 = self.attention_block_2(query=attention_out_1,
                                                 value=encoder_out,
                                                 key=encoder_out,
                                                 attention_mask=padd_mask,
                                                 )

        attention_out_2 = self.layer_norm_2(attention_out_1 + attention_out_2)

        dense_block_out = self.dense_block(attention_out_2)
        return self.layer_norm_3(attention_out_2 + dense_block_out)

    def get_config(self):
        conf = super().get_config()
        conf.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "multi_head_num": self.multi_head_num,
        })

        return conf
