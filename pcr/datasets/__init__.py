from __future__ import absolute_import

from .market1501 import Market1501
from .dukemtmc import DukeMTMC


__factory = {
    'market1501': Market1501,
    'dukemtmc-reid': DukeMTMC,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name, one of 'market1501', 'dukemtmc-reid'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
