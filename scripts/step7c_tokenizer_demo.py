"""Step 7c: what a naive tokenizer does to "401(k)" and friends, vs the deliberate one.

Run:  .venv/bin/python -m scripts.step7c_tokenizer_demo
"""
import re

from src.rag.keyword_index import tokenize

WORDS_ONLY = re.compile(r"\w+")   # the usual first attempt: "split on anything that is not a letter or digit"

samples = ["401(k) match", "S&P 500 index", "tax-free bonds", "the investor's expense ratio is 0.03", "what is HHI"]
for text in samples:
    print(f"text:           {text}")
    print(f"  naive split():  {text.lower().split()}")
    print(f"  naive \\w+:      {WORDS_ONLY.findall(text.lower())}")
    print(f"  deliberate:     {tokenize(text)}\n")
