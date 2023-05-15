import random

import tensorflow as tf
from keras import layers
from tensorflow import keras
from keras.optimizers import RMSprop, Adam
from keras.activations import relu
import numpy as np
import time

__author__ = "Aldy"

""" Here are hyper params. """

prep_train = 'C:/Dev/ds/wp_prep_01_train.txt'
prep_valid = 'C:/Dev/ds/wp_prep_01_valid.txt'
prep_test = 'C:/Dev/ds/wp_prep_01_test.txt'

vocab_size = 15000
seq_len = 32

embed_dim = 64
dense_dim = 512
multi_head_num = 2

conf_epoch = 3
conf_batch_size = 64
conf_dropout = 0.2
conf_lr = 0.01
conf_optimizer = RMSprop(learning_rate=conf_lr)
conf_optimizer_adam = Adam()
conf_activation = relu

""" Here are layers. """


class PosEmbedding(layers.Layer):

    def __init__(self, seq_len, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.term_embed = layers.Embedding(input_dim=in_dim, output_dim=out_dim)
        self.pos_embed = layers.Embedding(input_dim=seq_len, output_dim=out_dim)
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


class TDecoder(layers.Layer):
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


start_time = time.time()

ds_train_path = prep_train
ds_valid_path = prep_valid
ds_test_path = prep_test

with open(ds_train_path, encoding='UTF-8') as ds_train:
    lines_train = ds_train.read().split('\n')
ds_train.close()

with open(ds_valid_path, encoding='UTF-8') as ds_valid:
    lines_valid = ds_valid.read().split('\n')
ds_valid.close()

with open(ds_test_path, encoding='UTF-8') as ds_test:
    lines_test = ds_test.read().split('\n')
ds_test.close()


def get_as_pair(l):
    p1, p2 = l.split('\t')
    return p1, p2


pairs_train = list(map(get_as_pair, list(filter(None, lines_train))))
pairs_valid = list(map(get_as_pair, list(filter(None, lines_valid))))
pairs_test = list(map(get_as_pair, list(filter(None, lines_test))))

print(len(pairs_train))
print(len(pairs_valid))
print(len(pairs_test))

print(time.time() - start_time)
start_time = time.time()

src_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=seq_len,
)

tar_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=seq_len + 1,
)

print("cost(.one) is " + str(time.time() - start_time))
start_time = time.time()

train_src_seqs = [p[0] for p in pairs_train]
train_tar_seqs = [p[1] for p in pairs_train]

print("cost(.two) is " + str(time.time() - start_time))
start_time = time.time()

src_vectorization.adapt(train_src_seqs)
tar_vectorization.adapt(train_tar_seqs)

print("cost(.three) is " + str(time.time() - start_time))
start_time = time.time()


def format_dataset(src, tar):
    src = src_vectorization(src)
    tar = tar_vectorization(tar)
    return ({
                "source": src,
                "target": tar[:, :-1],
            }, tar[:, 1:])


print("cost(.four) is " + str(time.time() - start_time))
start_time = time.time()


def get_as_dataset(pairs):
    src_text, tar_text = zip(*pairs)
    src_text = list(src_text)
    tar_text = list(tar_text)
    dataset = tf.data.Dataset.from_tensor_slices((src_text, tar_text))
    dataset = dataset.batch(conf_batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=2)
    return dataset.shuffle(2048).prefetch(16).cache()


print("cost(.five) is " + str(time.time() - start_time))
start_time = time.time()

train_dataset = get_as_dataset(pairs_train)
valid_dataset = get_as_dataset(pairs_valid)
print("cost(.six) is " + str(time.time() - start_time))
start_time = time.time()

for inputs, targets in train_dataset.take(1):
    print(f"inputs['source'].shape: {inputs['source'].shape}")
    print(f"inputs['target'].shape: {inputs['target'].shape}")
    print(f"target.shape: {targets.shape}")

encoder_in = keras.Input(shape=(None,), dtype="int64", name='source')
print(encoder_in.shape)
forward = PosEmbedding(seq_len, vocab_size, embed_dim)(encoder_in)
print(forward.shape)
encoder_out = TEncoder(embed_dim, dense_dim, multi_head_num)(forward)
print(encoder_out.shape)
decoder_in = keras.Input(shape=(None,), dtype="int64", name='target')
forward = PosEmbedding(seq_len, vocab_size, embed_dim)(decoder_in)
forward = TDecoder(embed_dim, dense_dim, multi_head_num)(forward, encoder_out)
forward = layers.Dropout(conf_dropout)(forward)
decoder_out = layers.Dense(vocab_size, activation="softmax")(forward)

print("cost(.seven) is " + str(time.time() - start_time))
start_time = time.time()

tsfm = keras.Model([encoder_in, decoder_in], decoder_out)
tsfm.compile(optimizer=conf_optimizer,
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])

print("cost(.eight) is " + str(time.time() - start_time))
start_time = time.time()

tsfm.fit(train_dataset, epochs=conf_epoch, validation_data=valid_dataset)

print("cost(.nine) is " + str(time.time() - start_time))
start_time = time.time()

target_vocab = tar_vectorization.get_vocabulary()
target_index = dict(zip(range(len(target_vocab)), target_vocab))
max_target_seq_len = seq_len


def decode_seq(input_seq):
    tokenized_src = src_vectorization([input_seq])
    decoded_seq = '[SOS]'
    for i in range(max_target_seq_len):
        tokenized_tar = tar_vectorization([decoded_seq])[:, :-1]
        pred = tsfm([tokenized_src, tokenized_tar])
        sampled_token_index = np.argmax(pred[0, i, :])
        sampled_token = target_index[sampled_token_index]
        decoded_seq += ' ' + sampled_token
        if sampled_token == '[EOS]':
            break
    return decoded_seq


for _ in range(20):
    test_pair = random.choice(pairs_test)
    test_in = test_pair[0]
    test_out = test_pair[1]
    print('---')
    print('test_in: ' + test_in)
    print('test out: ' + test_out)
    print('test pred: ' + decode_seq(test_in))
