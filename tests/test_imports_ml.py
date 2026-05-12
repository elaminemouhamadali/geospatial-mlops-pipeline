# tests/test_imports_ml.py

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")


def test_ml_optional_imports():
    import torch
    import torchvision
    import transformers

    assert torch is not None
    assert torchvision is not None
    assert transformers is not None
