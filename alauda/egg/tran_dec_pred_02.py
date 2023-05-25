import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy, AUC
from keras.optimizers import Adam
from tensorflow import keras

__author__ = "Aldy"

""" Here are hyper params. """

train_file = 'C:/Dev/ds/lyrics/eng_lyrics_train_02.txt'
valid_file = 'C:/Dev/ds/lyrics/eng_lyrics_valid_02.txt'
test_file = 'C:/Dev/ds/lyrics/eng_lyrics_test_02.txt'

pd_train = pd.read_csv(train_file)
print(pd_train.shape)
pd_valid = pd.read_csv(valid_file)
print(pd_valid.shape)
pd_test = pd.read_csv(test_file)
print(pd_test.shape)

num_classes = len(pd.unique(pd_train['age']))
print(' *** multi-classes number ', num_classes)

chosen_cols = ['lyrics', 'age']

pairs_train = pd_train[chosen_cols].values.tolist()
pairs_valid = pd_valid[chosen_cols].values.tolist()
pairs_test = pd_test[chosen_cols].values.tolist()
print(' train pairs len ', len(pairs_train))
print(' valid pairs len ', len(pairs_valid))
print(' test pairs len ', len(pairs_test))

conf_vocab_size = 20000
conf_seq_len = 100

conf_embed_dim = 64
conf_dense_dim = 64
conf_multi_head_num = 8

conf_epoch = 5
conf_batch_size = 128
conf_lr = 0.01
conf_optimizer = Adam()
conf_activation = 'gelu'
conf_dropout = 0.4

conf_gen_len = 25
conf_print_freq = 1

""" Here are layers. """


class PosEmbedding(layers.Layer):

    def __init__(self, seq_len, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.term_embed = layers.Embedding(input_dim=in_dim,
                                           output_dim=out_dim)
        self.pos_embed = layers.Embedding(input_dim=seq_len,
                                          output_dim=out_dim)
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.out_dim = out_dim

    def call(self, inputs):
        len_ = tf.shape(inputs)[-1]
        pos = tf.range(start=0, limit=len_, delta=1)
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


class TEncoder(layers.Layer):

    def __init__(self,
                 embed_dim,
                 dense_dim,
                 multi_head_num=2,
                 activation_func="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.multi_head_num = multi_head_num
        self.attention_block = \
            layers.MultiHeadAttention(num_heads=multi_head_num,
                                      key_dim=embed_dim)
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


class TDecoder(layers.Layer):
    def __init__(self, embed_dim,
                 dense_dim,
                 multi_head_num=2,
                 activation_func="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.multi_head_num = multi_head_num
        self.attention_block_1 = \
            layers.MultiHeadAttention(multi_head_num,
                                      key_dim=embed_dim)
        self.attention_block_2 = \
            layers.MultiHeadAttention(multi_head_num,
                                      key_dim=embed_dim)
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
        multiples = tf.concat([tf.expand_dims(batch_size, -1),
                               tf.constant([1, 1],
                                           dtype=tf.int32)],
                              axis=0)
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


start_time = time.time()

inputs = keras.Input(shape=(None,),
                     dtype="int64")
forward_x = PosEmbedding(conf_seq_len,
                         conf_vocab_size,
                         conf_embed_dim)(inputs)
forward_x = TDecoder(conf_embed_dim,
                     conf_dense_dim,
                     conf_multi_head_num,
                     activation_func=conf_activation)(forward_x, forward_x)

forward_x = layers.GlobalMaxPooling1D()(forward_x)

# forward_x = layers.Embedding(input_dim=conf_embed_dim,
#                              output_dim=num_classes + 1)(forward_x)
forward_x = layers.Dropout(conf_dropout)(forward_x)

outputs = layers.Dense(num_classes,
                       activation="softmax")(forward_x)

print("cost(.01) is " + str(time.time() - start_time))
start_time = time.time()

gt_model = keras.Model(inputs, outputs)
gt_model.compile(optimizer=conf_optimizer,
                 loss="sparse_categorical_crossentropy",
                 metrics=['accuracy'], )

print("cost(.02) is " + str(time.time() - start_time))
start_time = time.time()

text_vectorization = layers.TextVectorization(
    max_tokens=conf_vocab_size,
    output_mode="int",
    output_sequence_length=conf_seq_len,
)

target_vectorization = layers.TextVectorization(
    max_tokens=num_classes + 2,
    output_mode="int",
    output_sequence_length=1,
)

train_texts = [p[0] for p in pairs_train]
train_targets = [p[1] for p in pairs_train]

text_vectorization.adapt(train_texts)
# target_vectorization.adapt(train_targets)

tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))


# target_index = dict(enumerate(target_vectorization.get_vocabulary()))
# print(target_index)


def format_dataset(src, tar):
    src = text_vectorization(src)
    tar = tar - 1
    return src, tar


def get_as_dataset(pairs):
    texts, targets = zip(*pairs)
    src = list(texts)
    tar = list(targets)
    ds = tf.data.Dataset.from_tensor_slices((src, tar))
    ds = ds.batch(conf_batch_size)
    ds = ds.map(format_dataset, num_parallel_calls=2)
    return ds.shuffle(2048).prefetch(16).cache()


train_dataset = get_as_dataset(pairs_train)
valid_dataset = get_as_dataset(pairs_valid)

print("cost(.03) is " + str(time.time() - start_time))
start_time = time.time()

print(" *** model summary ", gt_model.summary())


class TestCallback(keras.callbacks.Callback):

    def __init__(self,
                 test_pairs,
                 test_freq=1):
        super().__init__()
        self.test_pairs = test_pairs
        self.test_freq = test_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.test_freq != 0:
            return

        sampled_test_pairs = random.sample(self.test_pairs, 10)
        for pair in sampled_test_pairs:
            print('\n')
            print(' -- test with row lyric: ', pair[0])
            print(' -- test with row age: ', pair[1])
            tokenized_lyric = text_vectorization([pair[0]])
            prediction = self.model(tokenized_lyric)
            predicted_output = np.argmax(prediction)
            print(' -> predicted age: ', (predicted_output))


test_callback = TestCallback(
    test_pairs=pairs_test,
    test_freq=conf_print_freq,
)

gt_model.fit(train_dataset,
             epochs=conf_epoch,
             validation_data=valid_dataset,
             # callbacks=[test_callback])
             )
