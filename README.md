# HexapawnZero

Implementation of AlphaZero for the game of [Hexapawn](https://en.wikipedia.org/wiki/Hexapawn). I also wrote a [blog post](https://souvikshanku.github.io/blog/hexapawn/) on this!

## Example Usage

```bash
git clone https://github.com/souvikshanku/HexapawnZero.git
cd HexapawnZero

python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

# Train HexapawnZero
python3 self_play.py

# Play against the trained model
python3 play.py
```

## References

* [Neural Networks for Chess](https://arxiv.org/pdf/2209.01506.pdf)
* [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
* [A Simple Alpha(Go) Zero Tutorial](https://suragnair.github.io/posts/alphazero.html)
