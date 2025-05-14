from py_benchmark_functions import Function

from .builder import Builder
from .registry import Registry


# Functional API
def get_fn(fn_name: str, dims: int) -> Function:
    return Builder().function(fn_name).dims(dims).build()


def get_np_function(fn_name: str, dims: int) -> Function:
    return Builder().function(fn_name).numpy().dims(dims).build()


def get_tf_function(fn_name: str, dims: int) -> Function:
    return Builder().function(fn_name).tensorflow().dims(dims).build()


def available_backends() -> set[str]:
    return Registry.backends


def available_functions() -> list[str]:
    return Registry.functions


# Clear namescope
del Function
