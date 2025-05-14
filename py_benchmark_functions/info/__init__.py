"""Static metadata mapper."""

import functools
from math import pi

from py_benchmark_functions import Metadata


def repeat_coordinates(n: int, value: float) -> list[float]:
    return [value] * n


FunctionMetadata: dict[str, Metadata] = {
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
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Bohachevsky": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Brown": Metadata(
        default_search_space=(-1.0, 4.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "ChungReynolds": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Csendes": Metadata(
        default_search_space=(-1.0, 1.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Deb1": Metadata(
        default_search_space=(-1.0, 1.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Deb3": Metadata(
        default_search_space=(0.0, 1.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "DixonPrice": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Exponential": Metadata(
        default_search_space=(-1.0, 1.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Griewank": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Levy": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Mishra2": Metadata(
        default_search_space=(0.0, 1.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "PowellSum": Metadata(
        default_search_space=(-1.0, 1.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Qing": Metadata(
        default_search_space=(-500.0, 500.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Rastrigin": Metadata(
        default_search_space=(-5.12, 5.12),
        references=[
            "https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Rosenbrock": Metadata(
        default_search_space=(-30.0, 30.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "RotatedHyperEllipsoid": Metadata(
        default_search_space=(-30.0, 30.0),
        references=[
            "https://www.sfu.ca/~ssurjano/optimization.html",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Salomon": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Sargan": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "SumSquares": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Schwefel": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Schwefel12": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Schwefel222": Metadata(
        default_search_space=(-100.0, 100.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Schwefel223": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Schwefel226": Metadata(
        default_search_space=(-500.0, 500.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "SchumerSteiglitz": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Sphere": Metadata(
        default_search_space=(0.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "StrechedVSineWave": Metadata(
        default_search_space=(-10.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Trigonometric2": Metadata(
        default_search_space=(-500.0, 500.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Weierstrass": Metadata(
        default_search_space=(-0.5, 0.5),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Whitley": Metadata(
        default_search_space=(-10.24, 10.24),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "WWavy": Metadata(
        default_search_space=(-pi, pi),
        default_parameters=dict(k=10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
    "Zakharov": Metadata(
        default_search_space=(-5.0, 10.0),
        references=[
            "https://arxiv.org/abs/1308.4008",
        ],
        global_optimum=None,
        global_optimum_coordinates=None,
    ),
}
