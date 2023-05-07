from keras import layers
from tensorflow import keras

from src.alauda.common.transformer.naive_decoder import NaiveTransformerDecoder
from src.alauda.common.transformer.naive_encoder import NaiveTransformerEncoder
from src.alauda.common.transformer.pos_embed import PosEmbedding

vocab_size = 200000
seq_len = 600
embed_dim = 256
dense_dim = 32
multi_head_num = 2

conf_epoch = 10

train = ""
test = ""

encoder_in = keras.Input(shape=(None,), dtype="int32", name="naive_en")

forward = PosEmbedding(seq_len, vocab_size, embed_dim)(encoder_in)

encoder_out = NaiveTransformerEncoder(embed_dim, dense_dim, multi_head_num)(forward)

decoder_in = keras.Input(shape=(None,), dtype="int64", name="naive_de")

forward = PosEmbedding(seq_len, vocab_size, embed_dim)

forward = NaiveTransformerDecoder(embed_dim, dense_dim, multi_head_num)(forward, encoder_out)

forward = layers.Dropout(0.5)(forward)

decoder_out = layers.Dense(vocab_size, activation="softmax")(forward)

n_tsfm = keras.Model([encoder_in, decoder_in], decoder_out)

n_tsfm.compile(optimizer="rmsprop",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

n_tsfm.fit(train, epochs=conf_epoch, validation_data=test)
