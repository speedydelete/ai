
core_tokens = ['<reply>', '</reply>']

with open('words.txt') as file:
    words = []
    for word in file.read().split('\n'):
        word = word.split('#')[0]
        if word != '':
            words.append(word)
    

def tokenize(text: str) -> list[int]:
    out = []
    for word in text.split(' '):
        if word == '':
            out.append(160)
        elif word.isalpha() and word in words:
            out.append(words.index(word) + 1024)
        elif word in core_tokens:
            out.append(core_tokens.index(word) + 2)
        else:
            out.append(160)
            for char in word:
                num = ord(char)
                if num < 128:
                    out.append(num + 128)
                elif num < 65536:
                    out.append(num % 256 + 256)
                    out.append(num // 256 + 512)
                else:
                    out.append(num % 256 + 256)
                    out.append(num % 65535 // 256 + 512)
                    out.append(num // 65536 + 768)
    return out

def detokenize(data: list[int]) -> str:
    out = ''
    for token in data:
        if token == 1:
            out += '[unknown]'
        elif token >= 128 and token < 256:
            out += chr(token - 128)
        elif token >= 256 and token < 512:
            out += chr(token - 256)
        elif token >= 512 and token < 768:
            if len(out) > 0:
                out = out[:-1] + chr(ord(out[-1]) + (token - 512) * 256)
        elif token > 768 and token < 1024:
            if len(out) > 0:
                out = out[:-1] + chr(ord(out[-1]) + ((token - 768) % 17) * 65536)
        elif token >= 1024 and token < 1024 + len(words):
            out += ' ' + words[token - 1024]
    return out.encode('utf-16').decode('utf-16')
