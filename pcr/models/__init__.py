from __future__ import absolute_import

try:
    from .bpbreid_encoder import BPBReIDEncoder
    _import_error = None
except ImportError as e:  # torchreid (bpbreid's) not installed in this environment
    BPBReIDEncoder = None
    _import_error = e

__factory = {
    'bpbreid_encoder': BPBReIDEncoder,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """Create a model instance. Mirrors spcl/models/__init__.py's factory pattern."""
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    if __factory[name] is None:
        raise ImportError(
            "model '{}' requires torchreid (pip install -e /path/to/bpbreid): {}".format(name, _import_error))
    return __factory[name](*args, **kwargs)
