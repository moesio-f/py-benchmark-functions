<h1 align="center">
  <div style="display: flex; justify-content: space-between;">
  <a><img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/community/logos/python-logo-only.png" alt="Python" height="80"></a>
  <a><img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="NumPy" height="80"></a>
  <a><img src="https://upload.wikimedia.org/wikipedia/commons/a/ab/TensorFlow_logo.svg" alt="TensorFlow" height="80"></a>
  <a><img src="https://upload.wikimedia.org/wikipedia/commons/c/c6/PyTorch_logo_black.svg" alt="PyTorch" height="80"></a>
  </div>
  Python Benchmark Functions for Optimization
  <br>
</h1>
<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#benchmark-functions">Benchmark Functions</a>
</p>

> [!NOTE]  
> PyTorch support is underway ([#1](https://github.com/moesio-f/py-benchmark-functions/issues/1))!

[`py-benchamrk-functions`](https://github.com/moesio-f/py-benchmark-functions) is a simple library that provides benchmark functions for global optimization. It exposes implementations in major computing frameworks such as NumPy, TensorFlow and PyTorch. All implementations support `batch`-evaluation of coordinates, allowing for performatic evaluation of candidate solutions in the search space. The main goal of this library is to provide up-to-date implementations of multiple common benchmark functions in the scientific literature.

# Quick Start

## Installation

Start by installing the library using your preferred package manager:

```bash
python -m pip install --upgrade pip uv
python -m uv pip install py_benchmark_functions
```
```bash
# TensorFlow and Torch backend are optional and can be enable with:
python -m uv pip install py_benchmark_functions[tensorflow]
python -m uv pip install py_benchmark_functions[torch]
```

You can check if the library was correctly installed by running the following:

```python
import py_benchmark_functions as bf

print(bf.available_backends())
# Output: {'numpy', 'tensorflow'}

print(bf.available_functions())
# Output: ['Ackley', 'Alpine2', 'BentCigar', 'Bohachevsky', 'Brown', 'ChungReynolds', 'Csendes', 'Deb1', 'Deb3', 'DixonPrice', 'Exponential', 'Griewank', 'Levy', 'Mishra2', 'PowellSum', 'Qing', 'Rastrigin', 'Rosenbrock', 'RotatedHyperEllipsoid', 'Salomon', 'Sargan', 'SchumerSteiglitz', 'Schwefel', 'Schwefel12', 'Schwefel222', 'Schwefel223', 'Schwefel226', 'Sphere', 'StrechedVSineWave', 'SumSquares', 'Trigonometric2', 'WWavy', 'Weierstrass', 'Whitley', 'Zakharov']
```

## Instantiating and using Functions

The library is designed with the following entities:
- [`core.Function`](py_benchmBohachevskyark_functions/core/function.py): class that represents a benchmark function. An instance of this class represents an instance of the becnmark function for a given domain ([`core.Domain`](py_benchmBohachevskyark_functions/core/function.py)) and number of dimensions/coordinates.
- [`core.Transformation`](py_benchmBohachevskyark_functions/core/function.py): class that represents a _transformed_ (i.e., shifted, scaled, etc) function. It allows for programatically building new functions from existing ones.
- [`core.Metadata`](py_benchmBohachevskyark_functions/core/metadata.py): class thata represent _metadata_ about a given function (i.e., known global optima, default search space, default parameters, etc). A transformation inherits such metadata from the base function.

The benchmark functions can be instantiated in 3 ways:
1. Directly importing from `py_benchmark_functions.imp.{numpy,tensorflow,torch}` (e.g., `from py_benchmark_functions.imp.numpy import AckleyNumpy`);
```python
from py_benchmark_functions.imp.numpy import AckleyNumpy

fn = AckleyNumpy(dims=2)
print(fn.name, fn.domain)
# Output: Ackley Domain(min=[-35.0, -35.0], max=[35.0, 35.0])

print(fn.metadata)
Output: Metadata(default_search_space=(-35.0, 35.0), references=['https://arxiv.org/abs/1308.4008', 'https://www.sfu.ca/~ssurjano/optimization.html'], comments='', default_parameters={'a': 20.0, 'b': 0.2, 'c': 6.283185307179586}, global_optimum=0.0, global_optimum_coordinates=<...>)
```
2. Using the global `get_fn`, `get_np_function` or `get_tf_function` from `py_benchmark_functions`;
```python
import py_benchmark_functions as bf

fn = bf.get_fn("Zakharov", 2)
print(fn, type(fn))
# Output: Zakharov(domain=Domain(min=[-5.0, -5.0], max=[10.0, 10.0])) <class 'py_benchmark_functions.imp.numpy.ZakharovNumpy'>

fn1 = bf.get_np_function("Zakharov", 2)
print(fn1, type(fn1))
# Output: Zakharov(domain=Domain(min=[-5.0, -5.0], max=[10.0, 10.0])) <class 'py_benchmark_functions.imp.numpy.ZakharovNumpy'>

fn2 = bf.get_tf_function("Zakharov", 2)
print(fn2, type(fn2))
# Output: Zakharov(domain=Domain(min=[-5.0, -5.0], max=[10.0, 10.0])) <class 'py_benchmark_functions.imp.tensorflow.ZakharovTensorflow'>
```
3. Using the [`Builder`](py_benchmBohachevskyark_functions/factory/builder.py) class;
```python
from py_benchmark_functions import Builder

fn = Builder().function("Alpine2").dims(4).transform(vshift=1.0).tensorflow().build()
print(fn, type(fn))
# Output: Transformed(Alpine2) <class 'py_benchmark_functions.imp.tensorflow.TensorflowTransformation'>
```


Regardless of how you get an instance of a function, all of them define the `__call__` method, which allows them to be called directly. Every `__call__` receives an `x` as argument (for NumPy, `x` should be an `np.ndarray`, for Tensorflow a `tf.Tensor`, and for PyTorch a `torch.Tensor`). The shape of `x` can either be `(batch_size, dims)` or `(dims,)`, while the output is `(batch_size,)` or `()` (a scalar). Those properties are illustrated below:

```python
import py_benchmark_functions as bf
import numpy as np

fn = bf.get_fn("Ackley", 2)
x = np.array([0.0, 0.0], dtype=np.float32)

print(fn(x))
# Output: -9.536743e-07

x = np.expand_dims(x, axis=0)
print(x, fn(x))
# Output: [[0. 0.]] [-9.536743e-07]

x = np.repeat(x, 3, axis=0)
print(x, fn(x))
# Output:
# [[0. 0.]
# [0. 0.]
# [0. 0.]] [-9.536743e-07 -9.536743e-07 -9.536743e-07]
```

# Benchmark Functions

The following table lists the functions officially supported by the library. If you have any suggestion for new functions to add, we encourage you to open an issue or pull request.

| Function | NumPy | TensorFlow | PyTorch | Reference | Plot |
| --- | --- | --- | --- | --- | --- |
| Ackley | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008), [\[2\]](https://www.sfu.ca/~ssurjano/optimization.html) | ![](.github/images/plot-2d-Ackley.png) |
| Alpine 2 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Alpine2.png) |
| Bent Cigar | ✔️ | ✔️  | ⌛ | [\[3\]](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm) | ![](.github/images/plot-2d-BentCigar.png) |
| Bohachevsky | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Bohachevsky.png) |
| Brown | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Brown.png) |
| Chung Reynolds | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-ChungReynolds.png) |
| Csendes | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Csendes.png) |
| Deb 1 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008), [\[4\]](https://al-roomi.org/benchmarks/unconstrained/n-dimensions/231-deb-s-function-no-01) | ![](.github/images/plot-2d-Deb1.png) |
| Deb 3 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Deb3.png) |
| Dixon & Price | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008), [\[2\]](https://www.sfu.ca/~ssurjano/optimization.html) | ![](.github/images/plot-2d-DixonPrice.png) |
| Exponential | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Exponential.png) |
| Griewank | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Griewank.png) |
| Levy | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Levy.png) |
| Mishra 2 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008), [\[5\]](https://infinity77.net/global_optimization/test_functions_nd_M.html) | ![](.github/images/plot-2d-Mishra2.png) |
| Powell Sum | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-PowellSum.png) |
| Qing | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Qing.png) |
| Rastrigin | ✔️ | ✔️  | ⌛ | [\[3\]](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm) | ![](.github/images/plot-2d-Rastrigin.png) |
| Rosenbrock | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Rosenbrock.png) |
| Rotated Hyper-Ellipsoid | ✔️ | ✔️  | ⌛ | [\[2\]](https://www.sfu.ca/~ssurjano/optimization.html) | ![](.github/images/plot-2d-RotatedHyperEllipsoid.png) |
| Salomon | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Salomon.png) |
| Sargan | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Sargan.png) |
| Sum Squares | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-SumSquares.png) |
| Schwefel | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Schwefel.png) |
| Schwefel 1.2 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Schwefel12.png) |
| Schwefel 2.22 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Schwefel222.png) |
| Schwefel 2.23 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Schwefel223.png) |
| Schwefel 2.26 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Schwefel226.png) |
| Schumer Steiglitz | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-SchumerSteiglitz.png) |
| Sphere | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Sphere.png) |
| Streched V Sine Wave | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-StrechedVSineWave.png) |
| Trigonometric 2 | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Trigonometric2.png) |
| Weierstrass | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Weierstrass.png) |
| Whitley | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008), [\[5\]](https://infinity77.net/global_optimization/test_functions_nd_M.html) | ![](.github/images/plot-2d-Whitley.png) |
| W Wavy | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-WWavy.png) |
| Zakharov | ✔️ | ✔️  | ⌛ | [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008) | ![](.github/images/plot-2d-Zakharov.png) |

> [\[1\]](https://doi.org/10.48550/ARXIV.1308.4008): Jamil, M., & Yang, X.-S. (2013). A Literature Survey of Benchmark Functions For Global Optimization Problems. arXiv. https://doi.org/10.48550/ARXIV.1308.4008 
>
> [\[2\]](https://www.sfu.ca/~ssurjano/optimization.html): https://www.sfu.ca/~ssurjano/optimization.html
>
> [\[3\]](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm): Special Session & Competition on Single Objective Bound Constrained Optimization (CEC2021)
>
> [\[4\]](https://al-roomi.org/benchmarks/unconstrained/n-dimensions/231-deb-s-function-no-01): https://al-roomi.org/benchmarks/unconstrained/n-dimensions/231-deb-s-function-no-01
>
> [\[5\]](https://infinity77.net/global_optimization/test_functions_nd_M.html): https://infinity77.net/global_optimization/test_functions_nd_M.html
>
> _All the images can be generated using the [`Drawer`](py-benchmark-functions/blob/main/py_benchmark_functions/plot/drawer.py) utility._

