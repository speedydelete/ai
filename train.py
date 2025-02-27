
import sys
from ai import GPT

ai = GPT(
    vocab_size = 11000,
    maxlen = 1024,
    embed_dim = 512,
    blocks = 8,
    heads = 8,
    ff_dim = 1024,
)
ai.load('87m.keras')

with open('openwebtext.txt', 'r', encoding='utf-8') as file:
    text = file.read()

ai.train(text.split('\n\n')[0:10000], epochs=1, verbose=1)
# ai.train(text.split('\n\n'), epochs=10, verbose=1)

ai.save('87m.keras')
