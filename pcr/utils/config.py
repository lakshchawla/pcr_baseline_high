"""Minimal YAML config loader, used only by the CLIP training stages (examples/train_relational_prompts.py,
examples/train_relational_finetune.py) -- the rest of pcr2 stays argparse-only, per this repo's established
convention (see progress.md). Exposes the loaded config via dot-access (cfg.model.backbone)
rather than nested dict indexing, since these two scripts read many nested fields.
"""
import yaml


class ConfigNamespace(object):
    def __init__(self, d):
        for key, value in d.items():
            setattr(self, key, ConfigNamespace(value) if isinstance(value, dict) else value)

    def __repr__(self):
        return repr(vars(self))


def load_yaml_config(path):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return ConfigNamespace(raw)
