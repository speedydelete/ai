
from ai import GPT

ai = GPT(
    vocab_size = 11000,
    maxlen = 1024,
    embed_dim = 512,
    blocks = 8,
    heads = 8,
    ff_dim = 1024,
)
ai.compile()
ai.summary()
ai.save('87m.keras')
print('hi:', ai('hi', max_tokens = 64))
