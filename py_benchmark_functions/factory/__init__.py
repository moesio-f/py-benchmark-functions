from typing import List, Set

from py_benchmark_functions import Function
from py_benchmark_functions.factory.builder import Builder
from py_benchmark_functions.factory.registry import Registry


# Functional API
def get_fn(fn_name: str, dims: int) -> Function:
    return Builder().function(fn_name).dims(dims).build()


def get_np_function(fn_name: str, dims: int) -> Function:
    return Builder().function(fn_name).numpy().dims(dims).build()


def get_tf_function(fn_name: str, dims: int) -> Function:
    return Builder().function(fn_name).tensorflow().dims(dims).build()


def get_torch_function(fn_name: str, dims: int) -> Function:
    return Builder().function(fn_name).torch().dims(dims).build()


def available_backends() -> Set[str]:
    return Registry.backends


def available_functions() -> List[str]:
    return Registry.functions
