import struct
import requests

shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" # for tinyshakespeare as an example.
response = requests.get(shakespeare_url)
text = response.text

chars = ''.join(sorted(set(text)))
assert len(chars) == 65, f"Expected 65 unique characters, got {len(chars)}"

with open("tokenizer.bin", "wb") as f:
    max_token_length = 1
    f.write(struct.pack("i", max_token_length))
    for i, ch in enumerate(chars):
        score = 0.0 
        ch_bytes = ch.encode("utf-8")
        f.write(struct.pack("f", score))
        f.write(struct.pack("i", len(ch_bytes)))
        f.write(ch_bytes)

print("tokenizer.bin exported with 65 vocab size.")