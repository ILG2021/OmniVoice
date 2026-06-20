
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print(tokenizer.vocab_size)
print(len(tokenizer))

tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")
print(tokenizer.vocab_size)
print(len(tokenizer))
