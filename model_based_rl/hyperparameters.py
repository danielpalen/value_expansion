from ml_collections import ConfigDict, FrozenConfigDict
import numpy as np
import itertools
import hashlib


def _dict_flatten(in_dict, dict_out=None, parent_key=None, separator="."):
    if dict_out is None:
        dict_out = {}

    for k, v in in_dict.items():
        k = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            _dict_flatten(in_dict=v, dict_out=dict_out, parent_key=k)
            continue

        dict_out[k] = v
    return dict_out


def sample_hyper(seed, definition, hyper):
    np.random.seed(int(seed))

    flattened_dict = _dict_flatten(definition.to_dict())
    for key_i, val_i in flattened_dict.items():
        flattened_dict[key_i] = val_i[np.random.randint(0, len(val_i))]

    updated_hyper = ConfigDict(hyper)
    updated_hyper.update_from_flattened_dict(flattened_dict)
    return FrozenConfigDict(updated_hyper)


def select_hyper(idx, definition, hyper):
    flattened_dict = _dict_flatten(definition.to_dict())
    for key_i, val_i in flattened_dict.items():
        flattened_dict[key_i] = val_i[idx]

    updated_hyper = ConfigDict(hyper)
    updated_hyper.update_from_flattened_dict(flattened_dict)
    return FrozenConfigDict(updated_hyper)


def cartesian_product(definition):
    flattened_dict = _dict_flatten(definition.to_dict())
    keys, values = zip(*flattened_dict.items())
    grid_values = zip(*itertools.product(*values))
    updated_dict = dict(zip(keys, grid_values))

    grid_defintion = ConfigDict(definition)
    grid_defintion.update_from_flattened_dict(updated_dict)
    return grid_defintion


def compute_hash(hyper):
    norm_hyper = ConfigDict(hyper)
    norm_hyper.hash = ""
    norm_hyper.start_time = ""
    norm_hyper.env_name = ""
    norm_hyper.num_timesteps = 0
    norm_hyper.log_frequency = 0
    norm_hyper.seed = 0

    return hashlib.md5(str(norm_hyper).encode()).hexdigest()
