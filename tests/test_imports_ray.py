# tests/test_imports_ray.py

import pytest

pytest.importorskip("ray")


def test_ray_optional_imports():
    import ray

    assert ray is not None
