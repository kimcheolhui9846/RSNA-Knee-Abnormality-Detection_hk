import importlib


def test_src_packages_importable() -> None:
    for name in ("src", "src.data", "src.models"):
        importlib.import_module(name)
