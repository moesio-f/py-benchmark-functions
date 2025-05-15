import pytest

from py_benchmark_functions.factory import Registry
from py_benchmark_functions.info import FunctionMetadata


def test_all_backends_for_all_functions():
    for backend in Registry.backends:
        for fn in Registry.functions:
            assert Registry[fn, backend] is not None


def test_registry_and_info_same_functions():
    assert set(FunctionMetadata) == set(Registry.functions)


@pytest.mark.parametrize("fn_name", Registry.functions)
def test_not_missing_info(fn_name: str):
    metadata = FunctionMetadata.get(fn_name)
    assert metadata.global_optimum is not None
    assert metadata.global_optimum_coordinates is not None
