"""Metadata object about benchmark
functions.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Metadata:
    """Benchmark function metadata.

    Attributes:
        default_search_space (tuple[float, float]): search space bounds employed by
            the literature.
        references (list[str]): list of references with function definition.
        comments (str): extra information/comments about this function.
        default_parameters (dict[str, float], None): default values any set of
            parameters required by the function.
        global_optimum (float, Callable[[int], float], None): known global optimum (if any). Some
            functions global optimum depends on the number of dimensions.
        global_optimum_coordinates (Callable[[int], list[float]], None): function
            that returns the coordinates of global optimum given the number
            of dimensions as argument.
    """

    default_search_space: tuple[float, float]
    references: list[str]
    comments: str = ""
    default_parameters: dict[str, float] | None = None
    global_optimum: float | Callable[[int], float] | None = None
    global_optimum_coordinates: Callable[[int], list[float]] | None = None

    def concrete_optimum(self, dims: int) -> float:
        if self.global_optimum is None:
            return

        if isinstance(self.global_optimum, Callable):
            return self.global_optimum(dims)

        return self.global_optimum

    def concrete_otimum_coordinates(self, dims: int) -> list[float]:
        if self.global_optimum_coordinates is None:
            return

        return self.global_optimum_coordinates(dims)
