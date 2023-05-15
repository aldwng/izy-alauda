import tensorflow as tf
from keras import layers
from tensorflow import keras
import time

from alauda.common.const import prep_train, prep_valid, prep_test, seq_len, embed_dim, vocab_size, dense_dim, \
    multi_head_num, batch_size, conf_epoch
from alauda.common.transformer.naive_decoder import NaiveTransformerDecoder
from alauda.common.transformer.naive_encoder import NaiveTransformerEncoder
from alauda.common.transformer.pos_embed import PosEmbedding
from alauda.common.utils import get_as_pair

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

print("break.00")
print(time.time() - start_time)
start_time = time.time()

train_src_seqs = [p[0] for p in pairs_train]
train_tar_seqs = [p[1] for p in pairs_train]

print("break.01")
print(time.time() - start_time)
start_time = time.time()

src_vectorization.adapt(train_src_seqs)
tar_vectorization.adapt(train_tar_seqs)

print("break.1")
print(time.time() - start_time)
start_time = time.time()


def format_dataset(src, tar):
    src = src_vectorization(src)
    tar = tar_vectorization(tar)
    return ({
                "source": src,
                "target": tar[:, :-1],
            }, tar[:, 1:])


print("break.2")
print(time.time() - start_time)
start_time = time.time()


def get_as_dataset(pairs):
    src_text, tar_text = zip(*pairs)
    src_text = list(src_text)
    tar_text = list(tar_text)
    dataset = tf.data.Dataset.from_tensor_slices((src_text, tar_text))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=2)
    return dataset.shuffle(2048).prefetch(16).cache()


print("break.3")
print(time.time() - start_time)
start_time = time.time()

train_dataset = get_as_dataset(pairs_train)
valid_dataset = get_as_dataset(pairs_valid)
print("break.4")
print(time.time() - start_time)
start_time = time.time()

for inputs, targets in train_dataset.take(1):
    print(f"inputs['source'].shape: {inputs['source'].shape}")
    print(f"inputs['target'].shape: {inputs['target'].shape}")
    print(f"target.shape: {targets.shape}")

print("break.5")
print(time.time() - start_time)
start_time = time.time()

encoder_in = keras.Input(shape=(None,), dtype="int64", name='source')
print(encoder_in.shape)
forward = PosEmbedding(seq_len, vocab_size, embed_dim)(encoder_in)
print(forward.shape)
encoder_out = NaiveTransformerEncoder(embed_dim, dense_dim, multi_head_num)(forward)
print(encoder_out.shape)
decoder_in = keras.Input(shape=(None,), dtype="int64", name='target')
forward = PosEmbedding(seq_len, vocab_size, embed_dim)(decoder_in)
forward = NaiveTransformerDecoder(embed_dim, dense_dim, multi_head_num)(forward, encoder_out)
forward = layers.Dropout(0.5)(forward)
decoder_out = layers.Dense(vocab_size, activation="softmax")(forward)

print("break.6")
print(time.time() - start_time)
start_time = time.time()

n_tsfm = keras.Model([encoder_in, decoder_in], decoder_out)
n_tsfm.compile(optimizer="rmsprop",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

print("break.7")
print(time.time() - start_time)
start_time = time.time()

n_tsfm.fit(train_dataset, epochs=conf_epoch, validation_data=valid_dataset)

print("break.8")
print(time.time() - start_time)
start_time = time.time()
