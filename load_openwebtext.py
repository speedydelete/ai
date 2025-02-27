
from datasets import load_dataset

dataset = load_dataset('stas/openwebtext-10k', split='train')

text = ''
for item in dataset:
    text += item['text'] # type: ignore

with open('openwebtext.txt', 'w', encoding='utf-8') as file:
    file.write(text)
