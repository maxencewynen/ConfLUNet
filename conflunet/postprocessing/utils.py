import numpy as np


def convert_types(obj):
    # Convert all np.int32 types to standard python int for json dumping
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.float32)  or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types(i) for i in obj]
    return obj