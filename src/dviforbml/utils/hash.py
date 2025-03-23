import hashlib
import inspect
import json


def get_object_hash(obj: object) -> str:
    class_name = obj.__class__.__name__

    # Get __init__ parameters (excluding 'self')
    init_params = inspect.signature(obj.__class__.__init__).parameters
    init_param_names = [p for p in init_params if p != "self"]

    # Extract only the explicitly passed parameters
    params = {k: str(getattr(obj, k, None)) for k in init_param_names}

    # Serialize and hash the data
    data_str = json.dumps({"class": class_name, "params": params}, sort_keys=True)

    return hashlib.md5(data_str.encode()).hexdigest()


def get_var_name(var: object) -> str | None:
    """Returns the variable name as a string."""
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else frame  # Get the caller's frame

    if frame is None:
        return None

    for name, value in frame.f_locals.items():  # Iterate over local variables
        if value is var:
            return name

    return None  # Return None if the variable name can't be found
