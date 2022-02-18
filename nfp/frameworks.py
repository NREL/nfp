try:
    # First check if we're in a ray program and need a soft tf import
    from ray.rllib.utils.framework import try_import_tf  # type: ignore

    tf1, tf, tfv = try_import_tf()
    assert tfv == 2, "nfp requires tensorflow 2"

except ImportError:

    # Next try a regular tf import
    try:
        import tensorflow as tf

    except ImportError:
        tf = None
