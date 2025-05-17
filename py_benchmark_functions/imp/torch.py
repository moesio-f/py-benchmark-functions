from functools import cached_property
from math import e, pi
from typing import List, Tuple, Union

import torch

from py_benchmark_functions import config, core
from py_benchmark_functions.info import FunctionMetadata


class _TorchMixin:
    @property
    def device(self) -> torch.device:
        return self._device

    def set_device(self, device: torch.device):
        """Select the device to be used by the
        function. Inputs are moved to this device
        and outputs are on this device.

        Args:
            device (torch.device): device to use.
        """
        self._device = device

    def grads(self, x: torch.Tensor) -> torch.Tensor:
        grads, _ = self.grads_at(x)
        return grads

    def grads_at(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure that this tensor will be
        #   watched by autograd
        requires_grad = x.requires_grad
        x.requires_grad_(True)

        # Run function
        y = self(x)

        # Initialize grads for x
        y_for_grads = y.sum()
        y_for_grads.backward()

        # Store the x gradients and
        #   clear other grads
        grad = x.grad
        y = y.detach()

        # Restore initial state
        if not requires_grad:
            x.grad = None
            x.requires_grad_(False)

        # Return grad, f(x) pair
        return grad, y

    @cached_property
    def _domain_as_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            torch.tensor(v, dtype=self._dtype)
            for v in [self.domain.min, self.domain.max]
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Guarantee expected dtype
        x = x.type(self._dtype)

        # Select appropriate device to use
        device = x.device if self._device is None else self._device
        x = x.to(device)

        # Maybe check input shape
        if config.CHECK_INPUT_SHAPE:
            has_compatible_shape = len(x.shape) <= 2 and x.shape[-1] == self.dims
            if not has_compatible_shape:
                raise ValueError(
                    f"Incompatible shape for function {self.name} "
                    f"with {self.dims} dims: {x.shape}"
                )

        # Maybe check domain
        if config.CHECK_INPUT_DOMAIN:
            mi, ma = tuple(map(lambda v: v.to(device), self._domain_as_tensor))
            all_in_domain = (x >= mi).all() and (x <= ma).all()
            if not all_in_domain:
                if not config.COERCE_INPUT_TO_DOMAIN:
                    raise ValueError(
                        "Input values are out of bound for function "
                        f"{self.name} with domain {self.domain}."
                    )
                else:
                    x = x.clip(min=mi, max=ma)

        # Functions should return on the same device
        #   as `x`
        return self._fn(x).type(self._dtype)


class TorchFunction(_TorchMixin, core.Function):
    def __init__(
        self,
        dims: int,
        domain_min: Union[float, List[float]] = None,
        domain_max: Union[float, List[float]] = None,
        domain: core.Domain = None,
        dtype=torch.float32,
        device=None,
    ):
        if domain is None and domain_min is None:
            domain_min, domain_max = self.metadata.default_search_space
        super().__init__(dims, domain_min, domain_max, domain)
        self._dtype = dtype
        self._device = device

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Torch", "")

    @property
    def metadata(self) -> core.Metadata:
        return FunctionMetadata[self.name]


class TorchTransformation(_TorchMixin, core.Transformation):
    def __init__(
        self,
        fn: core.Function,
        vshift: float = 0.0,
        hshift: Union[float, List[float]] = 0.0,
        outer_scale: float = 1.0,
        inner_scale: Union[float, List[float]] = 1.0,
        has_same_domain: bool = False,
        dtype=torch.float32,
        device=None,
    ):
        is_compatible = isinstance(fn, TorchFunction) or isinstance(
            fn, TorchTransformation
        )
        assert is_compatible, "Only TorchFunctions are accepted."
        super().__init__(fn, vshift, hshift, outer_scale, inner_scale, has_same_domain)
        self._dtype = dtype
        self._device = device

    @cached_property
    def _params_as_tensor(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            torch.tensor(p, dtype=self._dtype)
            for p in [self.vshift, self.hshift, self.outer_scale, self.inner_scale]
        )

    def _fn(self, x: torch.Tensor) -> torch.Tensor:
        # Get parameters
        vs, hs, os, iscale = tuple(
            map(lambda v: v.to(x.device), self._params_as_tensor)
        )

        # Input transform
        x = iscale * x + hs

        # Apply function
        out = self.parent(x)

        # Apply output transforms
        out = os * out + vs

        return out


class AckleyTorch(TorchFunction):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain: core.Domain = None,
        dtype=torch.float32,
        device=None,
        a: float = None,
        b: float = None,
        c: float = None,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            dtype=dtype,
            device=device,
        )

        params = self.metadata.default_parameters
        self.a = params["a"] if a is None else a
        self.b = params["b"] if a is None else b
        self.c = params["c"] if a is None else c

    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1)
        sum2 = torch.sum(torch.cos(torch.multiply(x, self.c)), dim=-1)
        term1 = torch.multiply(
            torch.exp(torch.multiply(torch.sqrt(torch.divide(sum1, d)), -self.b)),
            -self.a,
        )
        term2 = torch.exp(torch.divide(sum2, d))
        result = term1 - term2 + self.a + e
        return result


class Alpine2Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        return torch.prod(torch.multiply(torch.sqrt(x), torch.sin(x)), dim=-1)


class BentCigarTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        indices = torch.arange(start=1, end=d, dtype=torch.int64, device=x.device)
        x0 = torch.squeeze(torch.index_select(x, index=indices[0] - 1, dim=-1))
        xi = torch.index_select(x, index=indices, dim=-1)
        return torch.pow(x0, 2) + torch.multiply(
            torch.sum(torch.pow(xi, 2), dim=-1), 1e6
        )


class BohachevskyTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        indices = torch.arange(start=0, end=2, dtype=torch.int64, device=x.device)
        x0 = torch.squeeze(torch.index_select(x, index=indices[0], dim=-1))
        x1 = torch.squeeze(torch.index_select(x, index=indices[1], dim=-1))
        out = (
            torch.pow(x0, 2)
            + torch.multiply(torch.pow(x1, 2), 2)
            - torch.multiply(torch.cos(3 * pi * x0), 0.3)
            - torch.multiply(torch.cos(4 * pi * x1), 0.4)
            + 0.7
        )

        # Maybe batch with size 1?
        out = maybe_batch(out, x.shape)

        return out


class BrownTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        indices = torch.arange(start=0, end=d, dtype=torch.int64, device=x.device)
        xi = torch.index_select(x, index=indices[:-1], dim=-1)
        xi1 = torch.index_select(x, index=indices[1:], dim=-1)

        xi_sq = torch.pow(xi, 2)
        xi1_sq = torch.pow(xi1, 2)

        return torch.sum(
            torch.pow(xi_sq, xi1_sq + 1) + torch.pow(xi1_sq, xi_sq + 1), dim=-1
        )


class ChungReynoldsTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        return torch.pow(torch.sum(torch.pow(x, 2), dim=-1), 2)


class CsendesTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        if torch.is_nonzero(torch.prod(x)):
            return torch.sum(
                torch.multiply(torch.pow(x, 6), 2 + torch.sin(torch.divide(1, x))),
                dim=-1,
            )
        return torch.sum(x * 0, dim=-1)


class Deb1Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        return -torch.divide(
            torch.sum(torch.pow(torch.sin(torch.multiply(x, 5 * pi)), 6), dim=-1), d
        )


class Deb3Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        return -torch.divide(
            torch.sum(
                torch.pow(
                    torch.sin(torch.multiply(torch.pow(x, 3 / 4) - 0.05, 5 * pi)), 6
                ),
                dim=-1,
            ),
            d,
        )


class DixonPriceTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        shape = x.shape
        x = torch.atleast_2d(x)
        d = x.shape[-1]
        x0 = x[:, 0]
        ii = torch.arange(start=2.0, end=d + 1, dtype=self._dtype, device=x.device)
        xi = x[:, 1:]
        xold = x[:, :-1]
        dixon_sum = ii * torch.pow(2 * torch.pow(xi, 2) - xold, 2)
        result = torch.pow(x0 - 1, 2) + torch.sum(dixon_sum, -1)
        result = maybe_batch(torch.squeeze(result), shape)
        return result


class ExponentialTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        return -torch.exp(torch.multiply(torch.sum(torch.pow(x, 2), dim=-1), -0.5))


class GriewankTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        initial_shape = x.shape
        x = torch.atleast_2d(x)
        shape = x.shape
        griewank_sum = torch.divide(torch.sum(torch.pow(x, 2), dim=-1), 4000)
        den = torch.arange(
            start=1, end=shape[-1] + 1, dtype=self._dtype, device=x.device
        )
        den = torch.repeat_interleave(den.unsqueeze(0), shape[0], dim=0)
        prod = torch.cos(torch.divide(x, torch.sqrt(den)))
        prod = torch.prod(prod, dim=-1)
        result = maybe_batch(torch.squeeze(griewank_sum - prod + 1), initial_shape)
        return result


class LevyTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        shape = x.shape
        x = torch.atleast_2d(x)
        d = x.shape[-1] - 1
        w = 1 + torch.divide(torch.subtract(x, 1), 4)

        term1 = torch.pow(torch.sin(pi * w[:, 0]), 2)
        wd = w[:, d]
        term3 = torch.pow(wd - 1, 2) * (1 + torch.pow(torch.sin(2 * pi * wd), 2))
        wi = w[:, 0:d]
        levy_sum = torch.sum(
            torch.pow((wi - 1), 2) * (1 + 10 * torch.pow(torch.sin(pi * wi + 1), 2)),
            dim=-1,
        )
        result = maybe_batch(torch.squeeze(term1 + levy_sum + term3), shape)
        return result


class Mishra2Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        indices = torch.arange(start=0, end=d, dtype=torch.int64, device=x.device)
        xi = torch.index_select(x, index=indices[:-1], dim=-1)
        xi1 = torch.index_select(x, index=indices[1:], dim=-1)
        xn = d - torch.sum(torch.multiply(xi + xi1, 0.5), dim=-1)
        return torch.pow(1 + xn, xn)


class PowellSumTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        indices = torch.arange(start=1, end=d + 1, dtype=self._dtype, device=x.device)
        return torch.sum(torch.pow(torch.abs(x), indices + 1), dim=-1)


class QingTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        indices = torch.arange(start=1, end=d + 1, dtype=self._dtype, device=x.device)
        return torch.sum(torch.pow(torch.pow(x, 2) - indices, 2), dim=-1)


class RastriginTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        return (10 * d) + torch.sum(
            torch.pow(x, 2) - (10 * torch.cos(torch.multiply(x, 2 * pi))), dim=-1
        )


class RosenbrockTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        shape = x.shape
        x = torch.atleast_2d(x)
        xi = x[:, :-1]
        xnext = x[:, 1:]
        result = torch.sum(
            100 * torch.pow(xnext - torch.pow(xi, 2), 2) + torch.pow(xi - 1, 2),
            dim=-1,
        )
        result = maybe_batch(torch.squeeze(result), shape)
        return result


class RotatedHyperEllipsoidTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        shape = x.shape
        x = torch.atleast_2d(x)
        d = x.shape[-1]
        mat = torch.repeat_interleave(x.unsqueeze(1), d, dim=1)
        matlow = mat.tril(0)
        inner = torch.sum(matlow**2, -1)
        result = torch.sum(inner, -1)
        result = maybe_batch(torch.squeeze(result), shape)
        return result


class SalomonTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        x_sqrt = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1))
        return (
            1 - torch.cos(torch.multiply(x_sqrt, 2 * pi)) + torch.multiply(x_sqrt, 0.1)
        )


class SarganTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        shape = x.shape
        d = shape[-1]
        inner_sum_axis = len(shape) - 1
        indices = torch.arange(start=1, end=d, dtype=torch.int64, device=x.device)
        xj = torch.index_select(x, index=indices, dim=-1).unsqueeze(-1)
        inner_x = x
        if inner_sum_axis > 0 and shape[0] > 1:
            inner_x = inner_x.unsqueeze(1)

        return torch.sum(
            torch.multiply(
                d,
                torch.pow(x, 2.0)
                + torch.multiply(torch.sum(inner_x * xj, dim=inner_sum_axis), 0.4),
            ),
            dim=-1,
        )


class SumSquaresTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        mul = torch.arange(
            start=1, end=x.shape[-1] + 1, dtype=self._dtype, device=x.device
        )
        return torch.sum(torch.multiply(torch.pow(x, 2), mul), dim=-1)


class SchumerSteiglitzTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        return torch.sum(torch.pow(x, 4), dim=-1)


class SchwefelTorch(TorchFunction):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain: core.Domain = None,
        a: float = None,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            dtype=dtype,
            device=device,
        )

        params = self.metadata.default_parameters
        self._a = params["a"] if a is None else a

    def _fn(self, x: torch.Tensor):
        return torch.pow(torch.sum(torch.pow(x, 2), dim=-1), self._a)


class Schwefel12Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        return torch.sum(torch.pow(torch.cumsum(x, dim=-1), 2), dim=-1)


class Schwefel222Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        x_abs = torch.abs(x)
        return torch.sum(x_abs, dim=-1) + torch.prod(x_abs, dim=-1)


class Schwefel223Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        return torch.sum(torch.pow(x, 10), dim=-1)


class Schwefel226Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        return -torch.divide(
            torch.sum(torch.multiply(x, torch.sin(torch.sqrt(torch.abs(x)))), dim=-1), d
        )


class SphereTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        return torch.sum(torch.pow(x, 2), dim=-1)


class StrechedVSineWaveTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        indices = torch.arange(start=0, end=d, dtype=torch.int64, device=x.device)
        xi_sqrd = torch.pow(torch.index_select(x, index=indices[:-1], dim=-1), 2)
        xi1_sqrd = torch.pow(torch.index_select(x, index=indices[1:], dim=-1), 2)
        sqrd_sum = xi1_sqrd + xi_sqrd
        return torch.sum(
            torch.multiply(
                torch.pow(sqrd_sum, 0.25),
                torch.pow(torch.sin(torch.multiply(torch.pow(sqrd_sum, 0.1), 50)), 2)
                + 0.1,
            ),
            dim=-1,
        )


class Trigonometric2Torch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        xi_squared = torch.pow(torch.subtract(x, 0.9), 2)
        x1_squared = torch.index_select(
            xi_squared,
            index=torch.tensor([0], dtype=torch.int64, device=x.device),
            dim=-1,
        )
        res_x = (
            torch.multiply(torch.pow(torch.sin(torch.multiply(xi_squared, 7)), 2), 8)
            + torch.multiply(torch.pow(torch.sin(torch.multiply(x1_squared, 14)), 2), 6)
            + xi_squared
        )
        return 1 + torch.sum(res_x, dim=-1)


class WWavyTorch(TorchFunction):
    def __init__(
        self,
        dims: int,
        domain_min=-pi,
        domain_max=pi,
        k: float = None,
        domain: core.Domain = None,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            dtype=dtype,
            device=device,
        )
        params = self.metadata.default_parameters
        self._k = params["k"] if k is None else k

    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        return 1 - torch.divide(
            torch.sum(
                torch.multiply(
                    torch.cos(torch.multiply(x, self._k)),
                    torch.exp(torch.divide(-torch.pow(x, 2), 2)),
                ),
                dim=-1,
            ),
            d,
        )


class WeierstrassTorch(TorchFunction):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain: core.Domain = None,
        dtype=torch.float32,
        device=None,
        a: float = None,
        b: float = None,
        kmax: int = None,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            dtype=dtype,
            device=device,
        )
        params = self.metadata.default_parameters
        self._a = params["a"] if a is None else a
        self._b = params["b"] if a is None else b
        self._kmax = params["kmax"] if a is None else kmax

    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        kindices = torch.arange(
            start=0, end=self._kmax + 1, dtype=self._dtype, device=x.device
        )

        #  Constants
        ak = torch.pow(self._a, kindices)
        bk = torch.pow(self._b, kindices)
        ak_cos_pi_bk = d * torch.sum(
            torch.multiply(ak, torch.cos(torch.multiply(bk, pi))), dim=-1
        )

        # Inner x
        inner_x = torch.add(x, 0.5).unsqueeze(-1)
        ak_cos_2pibk = torch.sum(ak * torch.cos(2 * pi * bk * inner_x), dim=-1)
        return torch.sum(ak_cos_2pibk, dim=-1) - ak_cos_pi_bk


class WhitleyTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        shape = x.shape
        d = shape[-1]
        has_batch = len(shape) > 1
        indices = torch.arange(start=0, end=d, dtype=torch.int64, device=x.device)
        if has_batch:
            acc = torch.zeros(shape[0], dtype=self._dtype, device=x.device)
        else:
            acc = torch.tensor(0.0, dtype=self._dtype, device=x.device)

        for j in indices:
            xj = torch.index_select(x, index=j, dim=-1)

            # Terms
            xi_sqrd = torch.pow(x, 2)
            xi_sqrd_minus_xj_all_sqrd = torch.pow(torch.subtract(xi_sqrd, xj), 2)
            one_minus_xj_all_sqrd = torch.pow(-torch.subtract(xj, 1), 2)
            hundred_composite = torch.multiply(xi_sqrd_minus_xj_all_sqrd, 100)

            # Terms
            t1 = torch.divide(
                torch.pow(
                    torch.add(
                        hundred_composite,
                        one_minus_xj_all_sqrd,
                    ),
                    2,
                ),
                4000,
            )
            t2 = torch.cos(
                torch.add(hundred_composite, one_minus_xj_all_sqrd),
            )

            # Accumulated value
            acc += torch.sum(t1 - t2 + 1, dim=-1)

        return acc


class ZakharovTorch(TorchFunction):
    def _fn(self, x: torch.Tensor):
        d = x.shape[-1]
        sum1 = torch.sum(torch.pow(x, 2), dim=-1)
        sum2 = torch.sum(
            torch.divide(
                torch.multiply(
                    x,
                    torch.arange(
                        start=1, end=(d + 1), dtype=self._dtype, device=x.device
                    ),
                ),
                2,
            ),
            dim=-1,
        )
        return sum1 + torch.pow(sum2, 2) + torch.pow(sum2, 4)


def maybe_batch(value: torch.Tensor, shape) -> torch.Tensor:
    if len(shape) > 1 and shape[0] == 1:
        return value.unsqueeze(0)
    return value
