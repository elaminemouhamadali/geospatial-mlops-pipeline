def test_geo_mlops_imports():
    import geo_mlops

    assert geo_mlops is not None


def test_core_dependency_imports():
    import mlflow
    import numpy
    import pandas
    import psutil
    import pydantic
    import pyproj
    import rasterio
    import tqdm
    import yaml

    assert mlflow is not None
    assert numpy is not None
    assert pandas is not None
    assert psutil is not None
    assert pydantic is not None
    assert pyproj is not None
    assert rasterio is not None
    assert tqdm is not None
    assert yaml is not None


if __name__ == "__main__":
    test_geo_mlops_imports()
    test_core_dependency_imports()
    print("base import tests OK")
