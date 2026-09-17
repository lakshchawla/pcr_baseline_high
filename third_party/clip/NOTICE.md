# Vendored dependency: CLIP (OpenAI)

Vendored from the `clip` package as installed via `pip install
git+https://github.com/openai/CLIP.git` (version 1.0), for local/internal modification and
debugging without a network dependency on GitHub at install time.

Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision"
(CLIP). ICML 2021. Released by OpenAI under the MIT License.

Only the text-tower-relevant modules are used by this repository
(`pcr/models/clip_text_encoder.py`); the visual encoder classes in `model.py` are vendored too
(they're part of the same file) but are never instantiated here.

Upstream: https://github.com/openai/CLIP
