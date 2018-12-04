
def merge_dict(base_dict, overwrite_dict):
    assert base_dict is not None
    if overwrite_dict is not None:
        return {**base_dict, **overwrite_dict}
    else:
        return base_dict
