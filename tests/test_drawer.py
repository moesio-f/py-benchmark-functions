import tempfile
from pathlib import Path

import os
import pytest

from py_benchmark_functions.factory import Registry
from py_benchmark_functions.plot import Drawer


@pytest.mark.skipif(
    "RUN_EXPENSIVE" not in os.environ,
    reason="Expensive tests should only be run if requested (RUN_EXPENSIVE environment variable).",
)
@pytest.mark.parametrize("fn_name", Registry.functions)
@pytest.mark.parametrize("backend", set([None]) | Registry.backends)
@pytest.mark.parametrize("dims", [2, 10])
def test_save_plot(fn_name: str, backend: str, dims: int):
    # Get function
    function = fn_name
    if backend:
        function = Registry[fn_name, backend](dims)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as dirname:
        # Create drawer
        drawer = Drawer(function)

        # Save in dir
        drawer.save(dirname)

        # Guarantee it was actually saved
        files = list(Path(dirname).glob("*"))
        assert len(files) == 1
        assert files[0].name.endswith(".png")

        # Clean-up the file
        files[0].unlink(missing_ok=False)
