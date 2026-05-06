import pytest


def test_all_imports_in_package():
    import symm_learning
    import symm_learning.nn
    import symm_learning.models

    # Collect all modules that define __all__
    modules_to_test = [
        symm_learning,
        symm_learning.nn,
        symm_learning.models,
    ]

    for mod in modules_to_test:
        if not hasattr(mod, "__all__"):
            continue

        for name in mod.__all__:
            # Check lazy import resolution
            try:
                getattr(mod, name)
            except AttributeError as e:
                pytest.fail(f"Could not import {name} from {mod.__name__}: {e}")

        # Also check _MODULE_MAP covers all declared exports if it exists
        if hasattr(mod, "_MODULE_MAP"):
            # Exclude __version__ as it's defined locally
            mod_all = set(mod.__all__) - {"__version__"}
            missing_in_map = mod_all - set(mod._MODULE_MAP.keys())
            missing_in_all = set(mod._MODULE_MAP.keys()) - mod_all

            if missing_in_map:
                pytest.fail(f"Module {mod.__name__} has items in __all__ missing from _MODULE_MAP: {missing_in_map}")
            if missing_in_all:
                pytest.fail(f"Module {mod.__name__} has items in _MODULE_MAP missing from __all__: {missing_in_all}")
