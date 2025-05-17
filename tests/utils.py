from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import torch

from py_benchmark_functions import Function, core
from py_benchmark_functions.imp import tensorflow as tff
from py_benchmark_functions.imp import torch as torchf


def batch_value(value, batch_size: int):
    if batch_size <= 0:
        return value

    if tf.is_tensor(value):
        return tf.repeat(tf.expand_dims(value, 0), batch_size, 0)

    if torch.is_tensor(value):
        return torch.repeat_interleave(value.unsqueeze(0), batch_size, dim=0)

    return np.repeat(np.expand_dims(value, 0), batch_size, 0)


def to_tensor_or_array(fn: Function, value: List[float]):
    if isinstance(fn, tff.TensorflowFunction):
        return tf.constant(value, dtype=tf.float32)

    if isinstance(fn, torchf.TorchFunction):
        return torch.tensor(value, dtype=torch.float32)

    return np.array(value, dtype=np.float32)


class EvaluationSamples:
    x_4d: List[float] = [1.0, 2.0, 3.0, 4.0]
    fx_4d: Dict[str, float] = {
        "Ackley": 8.43469444443746497,
        "Alpine2": -0.40033344730936005,
        "BentCigar": 29000001.0,
        "Bohachevsky": 9.6,
        "Brown": 1.6678281e16,
        "ChungReynolds": 900.0,
        "Csendes": 11.3658839392,
        "Deb1": -6.182844847431069e-87,
        "Deb3": -0.125,
        "DixonPrice": 4230.0,
        "Exponential": -0.135335283237,
        "Griewank": 1.00187037800320189,
        "Levy": 2.76397190019909811,
        "Mishra2": 2.0,
        "PowellSum": 4.0,
        "Qing": 184.0,
        "Rastrigin": 30.0,
        "Rosenbrock": 2705.0,
        "RotatedHyperEllipsoid": 50.0,
        "Salomon": 2.5375017928784365,
        "Sargan": 264.0,
        "SchumerSteiglitz": 354.0,
        "Schwefel12": 146.0,
        "Schwefel222": 34.0,
        "Schwefel223": 1108650.0,
        "Schwefel226": -2.353818129766789,
        "Schwefel": 43703.20448793846,
        "Sphere": 30.0,
        "StrechedVSineWave": 3.41314177672,
        "SumSquares": 100.0,
        "Trigonometric2": 31.645637672,
        "WWavy": 1.1130512151573806,
        "Weierstrass": 15.9999923706,
        "Whitley": 367532.562701,
        "Zakharov": 50880.0,
    }

    zeroes_x_4d: List[float] = [0.0, 0.0, 0.0, 0.0]
    zeroes_fx_4d: Dict[str, float] = {
        "Ackley": 0.0,
        "Alpine2": 0.0,
        "Bohachevsky": 0.0,
        "BentCigar": 0.0,
        "Brown": 0.0,
        "ChungReynolds": 0.0,
        "Csendes": 0.0,
        "Deb1": 0.0,
        "Deb3": -0.1249999850988388,
        "DixonPrice": 1.0,
        "Exponential": -1.0,
        "Griewank": 0.0,
        "Levy": 0.897533662350923467,
        "Mishra2": 625.0,
        "PowellSum": 0.0,
        "Qing": 30.0,
        "Rastrigin": 0.0,
        "Rosenbrock": 3.0,
        "RotatedHyperEllipsoid": 0.0,
        "Salomon": 0.0,
        "Sargan": 0.0,
        "SchumerSteiglitz": 0.0,
        "Schwefel12": 0.0,
        "Schwefel222": 0.0,
        "Schwefel223": 0.0,
        "Schwefel226": 0.0,
        "Schwefel": 0.0,
        "Sphere": 0.0,
        "StrechedVSineWave": 0.0,
        "SumSquares": 0.0,
        "Trigonometric2": 36.10124588012695,
        "WWavy": 0.0,
        "Weierstrass": 0.0,
        "Whitley": 7.359163106109763,
        "Zakharov": 0.0,
    }

    @classmethod
    def sample_eval(cls, function: core.Function, dims: int) -> Tuple[list, float]:
        name = function.name
        assert dims == 4
        assert name in cls.fx_4d

        return cls.x_4d, cls.fx_4d[name]

    @classmethod
    def sample_at_zeroes(cls, function: core.Function, dims: int) -> Tuple[list, float]:
        name = function.name
        assert dims == 4
        assert name in cls.fx_4d

        return cls.zeroes_x_4d, cls.zeroes_fx_4d[name]
