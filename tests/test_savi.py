"""Test the package."""


from importlib.util import find_spec


def test_importable():
    """Check the package is importable."""
    savi_spec = find_spec("savi")
    assert savi_spec is not None

    # Import it
    import savi

    assert savi is not None
