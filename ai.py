
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import ops
from tokenizer import tokenize, detokenize

keras.mixed_precision.set_global_policy('mixed_float16')


def causal_attention_mask(batch_size: int, n_dest: int, n_src: int, dtype: str):
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)

class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim: int, heads: int, ff_dim, rate: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation='relu'),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, 'bool')
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def build(self, input_shape):
        assert input_shape[-1] == self.embed_dim, f'Expected last dimension to be {self.embed_dim}, got {input_shape[-1]}'
        super().build(input_shape)


class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def build(self, input_shape):
        super().build(input_shape)


loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# def sample_from(logits):
#     logits, indices = ops.top_k(logits, k=10, sorted=True)
#     indices = np.asarray(indices).astype('int32')
#     preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
#     preds = np.asarray(preds).astype('float32')
#     return np.random.choice(indices, p=preds)

def sample_from(logits, temperature: float = 1.0):
    logits = (logits - np.max(logits)) / temperature
    preds = keras.activations.softmax(logits).numpy() # type: ignore
    preds = preds / np.sum(preds)
    return np.random.choice(np.arange(len(preds)), p=preds)

class GPT:

    def __init__(self, vocab_size: int | keras.Model, maxlen: int, embed_dim: int, blocks: int, heads: int, ff_dim: int) -> None:
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.blocks = blocks
        self.heads = heads
        self.ff_dim = ff_dim
        inputs = layers.Input(shape=(maxlen,), dtype='int32')
        x = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
        for _ in range(blocks):
            x = TransformerBlock(embed_dim, heads, ff_dim)(x)
        outputs = layers.Dense(vocab_size)(x)
        self.model = keras.Model(inputs=inputs, outputs=[outputs, x])
        self.vectorize_layer = layers.TextVectorization(vocab_size, output_mode='int', output_sequence_length=maxlen + 1)

    def compile(self) -> None:
        self.model.compile('adam', loss=[loss_function, None]) # type: ignore
    
    def summary(self) -> None:
        self.model.summary() # type: ignore

    def train(self, data: str | list[str], epochs: int = 25, verbose: int = 0) -> None:
        if (isinstance(data, str)):
            data = [data]
        if verbose > 0:
            print('tokenizing')
        sequences = []
        n = 0
        chars = 0
        token_count = 0
        for text in data:
            tokens = tokenize(text)
            while len(tokens) > 512:
                sequences.append(tokens[-512:])
                tokens = tokens[:-512]
            sequences.append(tokens)
            chars += len(text)
            token_count += len(tokens)
            if verbose > 0 and n % 1000 == 0 and n != 0:
                print(f'sequence {n} out of {len(data)} ({round(n/len(data)*100, 2)}%), tokenized, total: {chars} characters, {token_count} tokens')
            n += 1
        if verbose > 0:
            print(f'sequence {n} out of {len(data)} ({round(n/len(data)*100, 2)}%), tokenized, total: {chars} characters, {token_count} tokens')
        sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        x_train, y_train = sequences[:, :-1], sequences[:, 1:]
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=self.maxlen, padding='post')
        y_train = keras.preprocessing.sequence.pad_sequences(y_train, maxlen=self.maxlen, padding='post')
        self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=2) # type: ignore
        if verbose > 0:
            print('training complete')

    def __call__(self, text: str, temperature: float = 1.0, max_tokens: int | None = None) -> str:
        if max_tokens == None:
            max_tokens = self.maxlen
        start_tokens = tokenize(text)
        tokens = [x for x in start_tokens]
        tokens_generated = []
        while len(tokens) <= max_tokens:
            pad_len = self.maxlen - len(tokens)
            sample_index = len(tokens) - 1
            if pad_len < 0:
                x = tokens[:self.maxlen]
                sample_index = self.maxlen - 1
            elif pad_len > 0:
                x = tokens + [0] * pad_len
            else:
                x = tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0) # type: ignore
            sample_token = sample_from(y[0][sample_index], temperature)
            if sample_token == 0:
                break
            tokens_generated.append(sample_token)
            tokens.append(sample_token)
        return detokenize(tokens_generated)

    def save(self, filename: str):
        self.model.save(filename) # type: ignore

    def load(self, filename: str):
        self.model = keras.models.load_model(filename, {'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding})
