def test_ray_optional_imports():
    import ray

    assert ray is not None

if __name__ == "__main__":
    test_ray_optional_imports()
    print("ray import tests OK")