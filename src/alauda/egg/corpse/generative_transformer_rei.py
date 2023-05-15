import time

import numpy as np
import tensorflow as tf
from keras import callbacks
from keras import layers
from keras.activations import relu
from keras.optimizers import RMSprop
from tensorflow import keras

__author__ = "Aldy"

""" Here are hyper params. """

text_file_dir = 'C:/Dev/ds/war_peace/'
# text_file = 'C:/Dev/ds/war_peace/wp_prep_d_01.txt'
text_file = 'C:/Dev/ds/lyrics/eng_lyrics_00.txt'

vocab_size = 12000
seq_len = 50

embed_dim = 64
dense_dim = 128
multi_head_num = 4

conf_epoch = 20
conf_batch_size = 32
conf_lr = 0.01
conf_optimizer = RMSprop()
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

inputs = keras.Input(shape=(None,),
                     dtype="int64")
forward_x = PosEmbedding(seq_len,
                         vocab_size,
                         embed_dim)(inputs)
forward_x = TDecoder(embed_dim,
                     dense_dim,
                     multi_head_num)(forward_x, forward_x)
outputs = layers.Dense(vocab_size,
                       activation="softmax")(forward_x)

print("cost(.01) is " + str(time.time() - start_time))
start_time = time.time()

gt_model = keras.Model(inputs, outputs)
gt_model.compile(optimizer=conf_optimizer,
                 loss="sparse_categorical_crossentropy",
                 )

print("cost(.02) is " + str(time.time() - start_time))
start_time = time.time()

# dataset = keras.utils.text_dataset_from_directory(
#     directory=text_file,
#     labels=None,
#     label_mode=None,
#     batch_size=conf_batch_size
# )

with open(text_file, encoding='UTF-8') as txt_file:
    datalines = txt_file.read().split('\n')
txt_file.close()

datalines = list(filter(None, datalines))
print(' --- File been read lines ', len(datalines))


def get_as_dataset(dls):
    ds = tf.data.Dataset.from_tensor_slices(dls)
    ds = ds.batch(conf_batch_size)
    return ds.shuffle(2048).prefetch(16).cache()


dataset = get_as_dataset(datalines)

text_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=seq_len,
)

text_vectorization.adapt(dataset)


def prep_dataset(text_batch):
    vectorized_sequences = text_vectorization(text_batch)
    i = vectorized_sequences[:, :-1]
    j = vectorized_sequences[:, 1:]
    return i, j


decoder_dataset = dataset.map(prep_dataset, num_parallel_calls=2)

tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))


def stochastic_gen_next(predictions, temperature=0.5):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature
    p_exp = np.exp(predictions)
    predictions = p_exp / np.sum(p_exp)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)


class TextGenerator(callbacks.Callback):

    def __init__(self,
                 prompt,
                 gen_len,
                 model_input_len,
                 temperatures=(1.,),
                 print_freq=1):
        super().__init__()
        self.prompt = prompt
        self.gen_len = gen_len
        self.model_input_len = model_input_len
        self.temperatures = temperatures
        self.print_freq = print_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq != 0:
            return
        for temperature in self.temperatures:
            print('\n')
            print("--- Generating with temperature ", temperature)
            sentence = self.prompt
            for i in range(self.gen_len):
                tokenized_sentence = text_vectorization([sentence])
                predictions = self.model(tokenized_sentence)
                next_token = stochastic_gen_next(predictions[0, i, :], temperature)
                next_word = tokens_index[next_token]
                sentence += ' ' + next_word
            print(sentence)


prompt = "gold chain"

text_gen_callback = TextGenerator(
    prompt,
    gen_len=30,
    model_input_len=seq_len,
    temperatures=(0.1, 0.3, 0.5, 0.7, 1.,),
    print_freq=1,
)

print("cost(.03) is " + str(time.time() - start_time))
start_time = time.time()

gt_model.fit(decoder_dataset, epochs=conf_epoch, callbacks=[text_gen_callback])
