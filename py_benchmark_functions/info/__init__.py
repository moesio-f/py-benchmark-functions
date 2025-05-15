"""Static metadata mapper."""

import functools
from math import pi, sqrt
from typing import Callable, Dict, List

from py_benchmark_functions import Metadata


def repeat_coordinates(n: int, value: float) -> List[float]:
    return [value] * n


def coordinates_by_index_rule(n: int, rule: Callable[[int], float]) -> List[float]:
    return [rule(i) for i in range(n)]


FunctionMetadata: Dict[str, Metadata] = {
    "Ackley": Metadata(
        default_search_space=(-35.0, 35.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
            "https://www.sfu.ca/~ssurjano/optimization.html",
        ],
        default_parameters=dict(a=20.0, b=0.2, c=2.0 * pi),
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Alpine2": Metadata(
        default_search_space=(0.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=lambda d: 2.808**d,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=7.917),
    ),
    "BentCigar": Metadata(
        default_search_space=(0.0, 10.0),
        references=[
            "https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Bohachevsky": Metadata(
        default_search_space=(-100.0, 100.0),
        comments="Only first two coordinates have meaning.",
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Brown": Metadata(
        default_search_space=(-1.0, 4.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "ChungReynolds": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Csendes": Metadata(
        default_search_space=(-1.0, 1.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Deb1": Metadata(
        default_search_space=(-1.0, 1.0),
        comments="Has multiple global optimum coordinates.",
        references=[
            "https://arxiv.org/abs/1308.4008",
            "https://al-roomi.org/benchmarks/unconstrained/n-dimensions/231-deb-s-function-no-01",
        ],
        global_optimum=-1.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=pi / 2),
    ),
    "Deb3": Metadata(
        default_search_space=(0.0, 1.0),
        comments="Has multiple global optimum coordinates.",
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=-1.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=pi / 2),
    ),
    "DixonPrice": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
            "https://www.sfu.ca/~ssurjano/dixonpr.html",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(
            coordinates_by_index_rule, rule=lambda i: 2 ** -((2**i - 2) / (2**i))
        ),
    ),
    "Exponential": Metadata(
        default_search_space=(-1.0, 1.0),
        comments="The reference shows the wrong global optimum value (1.0 instead of -1.0).",
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=-1.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Griewank": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Levy": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=1.0),
    ),
    "Mishra2": Metadata(
        default_search_space=(0.0, 1.0),
        comments="Used definition by [2].",
        references=[
            "https://arxiv.org/abs/1308.4008",
            "https://infinity77.net/global_optimization/test_functions_nd_M.html",
        ],
        global_optimum=2.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=1.0),
    ),
    "PowellSum": Metadata(
        default_search_space=(-1.0, 1.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Qing": Metadata(
        default_search_space=(-500.0, 500.0),
        comments="Global minima also occurs at -sqrt(i).",
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(
            coordinates_by_index_rule, rule=lambda i: sqrt(i + 1)
        ),
    ),
    "Rastrigin": Metadata(
        default_search_space=(-5.12, 5.12),
        references=[
            "https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Rosenbrock": Metadata(
        default_search_space=(-30.0, 30.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=1.0),
    ),
    "RotatedHyperEllipsoid": Metadata(
        default_search_space=(-30.0, 30.0),
        references=[
            "https://www.sfu.ca/~ssurjano/optimization.html",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Salomon": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Sargan": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "SumSquares": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Schwefel": Metadata(
        default_search_space=(-100.0, 100.0),
        default_parameters=dict(a=pi),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Schwefel12": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Schwefel222": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Schwefel223": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Schwefel226": Metadata(
        default_search_space=(-500.0, 500.0),
        comments="Reference wrongly suggests global optimum equals to -418.983.",
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "SchumerSteiglitz": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Sphere": Metadata(
        default_search_space=(0.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "StrechedVSineWave": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Trigonometric2": Metadata(
        default_search_space=(-500.0, 500.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=1.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.9),
    ),
    "Weierstrass": Metadata(
        default_search_space=(-0.5, 0.5),
        default_parameters=dict(a=0.5, b=3.0, kmax=20),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Whitley": Metadata(
        default_search_space=(-10.24, 10.24),
        comments="Implements the [2] reference.",
        references=[
            "https://arxiv.org/abs/1308.4008",
            "https://infinity77.net/global_optimization/test_functions_nd_W.html",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=1.0),
    ),
    "WWavy": Metadata(
        default_search_space=(-pi, pi),
        default_parameters=dict(k=10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
    "Zakharov": Metadata(
        default_search_space=(-5.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=0.0,
        global_optimum_coordinates=functools.partial(repeat_coordinates, value=0.0),
    ),
}
