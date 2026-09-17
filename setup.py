from setuptools import setup, find_packages

# clip and torchreid (bpbreid) are vendored under third_party/ so `setup.py develop`/
# `pip install -e .` builds everything this repo needs from local source -- no separate
# `pip install -e /path/to/bpbreid` or `pip install git+https://github.com/openai/CLIP.git`
# step, and no dependency on those sibling repos existing on disk at all. package_dir maps
# each vendored package's real name (`clip`, `torchreid`) to its third_party/ subdirectory,
# so `import clip` / `import torchreid` resolve there directly, exactly as pcr/models/
# clip_text_encoder.py and pcr/models/bpbreid_encoder.py already expect.
setup(
    name='pcr',
    version='0.1.0',
    description='PCR: BPBReID part-based feature pooling trained under SpCL self-paced UDA',
    author='',
    packages=find_packages(include=['pcr', 'pcr.*']) + find_packages(where='third_party'),
    package_dir={
        'clip': 'third_party/clip',
        'torchreid': 'third_party/torchreid',
    },
    package_data={'clip': ['bpe_simple_vocab_16e6.txt.gz']},
    install_requires=[],
)
