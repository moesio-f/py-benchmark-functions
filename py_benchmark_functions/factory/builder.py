"""Function builder."""

from typing import List, Union

try:
    from typing import Self
except ImportError:
    # Self was added on Python 3.11
    from typing import TypeVar

    Self = TypeVar("Self", bound="Builder")

from py_benchmark_functions import Domain, Function, Transformation
from py_benchmark_functions.factory.registry import Registry


class Builder:
    """Programmatically way for building functions
    with multiple transformations or custom properties.
    """

    def __init__(self):
        self._dims = None
        self._base_domain = dict()
        self._base_backend = None
        self._base_fn_name = None
        self._transformations: List[dict] = []

    def function(self, fn_name: str) -> Self:
        """Select the base function to use.

        Args:
            fn_name (str): name of the base function
                as it appears on the registry.

        Returns:
            Builder: self.
        """
        assert fn_name in Registry
        if self._base_backend:
            assert (fn_name, self._base_backend) in Registry

        self._base_fn_name = fn_name
        return self

    def domain(
        self,
        domain_min: Union[float, List[float]] = None,
        domain_max: Union[float, List[float]] = None,
        domain: Domain = None,
    ):
        """Set the target domain for this function. If
            no domain is set, a default one is used.

        Args:
            domain_min (float, list[float]): minimum values for x.
            domain_max (float, list[float]): maximum values for x.
            domain (Domain): same as domain_{min,max} that only
                supports lists of floats (one <min, max> pair for
                dimensions).

        Returns:
            Builder: self.
        """
        self._base_domain = dict(
            domain_min=domain_min, domain_max=domain_max, domain=domain
        )
        return self

    def dims(self, dims: int) -> Self:
        """Set the number of dimensions for this
        function.

        Args:
            dims (int): number of dimensions.

        Returns:
            Builder: self.
        """
        assert dims > 1
        for t in self._transformations:
            for k in ["hshift", "inner_scale"]:
                v = t[k]
                if isinstance(v, list):
                    assert (
                        len(v) == self._dims
                    ), f"Dims ({self._dims}) don't match transformation paramenters ({len(v)})."

        self._dims = dims
        return self

    def numpy(self) -> Self:
        """Select numpy as the backend.

        Returns:
            Builder: self.
        """
        self._select_backend("numpy")
        return self

    def tensorflow(self) -> Self:
        """Select tensorflow as the backend. Must
            have installed the library with tensorflow
            support (see README).

        Returns:
            Builder: self.
        """
        self._select_backend("tensorflow")
        return self

    def torch(self) -> Self:
        """Select torch as the backend. Must
            have installed the library with torch
            support (see README).

        Returns:
            Builder: self.
        """
        self._select_backend("torch")
        return self

    def transform(
        self,
        vshift: float = 0.0,
        hshift: Union[float, List[float]] = 0.0,
        outer_scale: float = 1.0,
        inner_scale: Union[float, List[float]] = 1.0,
        has_same_domain: bool = False,
    ) -> Self:
        """Add a transformation for either the base
        function or the last transformation.

        Args: see core.Transformation.

        Returns:
            Builder: self.
        """
        assert (
            self._base_fn_name is not None
        ), "Select base function prior to applying transformations."

        if self._dims is not None:
            for v in (hshift, inner_scale):
                if isinstance(v, list):
                    assert (
                        len(v) == self._dims
                    ), f"Transformation parameters ({len(v)}) don't match dims ({self._dims})."

        # Maybe skip?
        if (
            vshift == 0.0
            and hshift == 0.0
            and outer_scale == 1.0
            and inner_scale == 1.0
        ):
            return self

        # Store transformations
        self._transformations.append(
            dict(
                vshift=vshift,
                hshift=hshift,
                outer_scale=outer_scale,
                inner_scale=inner_scale,
                has_same_domain=has_same_domain,
            )
        )

        return self

    def build(self) -> Function:
        """Builds the function represented
        by this builder.

        Returns:
            Function: concrete instance of the
                function with the selected backend
                and properties.
        """
        # Assertions
        assert self._dims is not None
        assert self._base_fn_name is not None

        # Initialize defaults
        if self._base_backend is None:
            self._select_backend("numpy")

        # Maybe remove nones
        domain_items = list(self._base_domain.items())
        for k, v in domain_items:
            if v is None:
                del self._base_domain[k]

        # Initialize base function
        fn = Registry.get(self._base_fn_name, self._base_backend)(
            self._dims, **self._base_domain
        )

        # Initialize transformations
        fn = self._apply_transformations(fn)

        return fn

    def _apply_transformations(self, fn: Function) -> Union[Function, Transformation]:
        # Maybe skip importing if not needed
        if not self._transformations:
            return fn

        # Select appropriate backend
        from py_benchmark_functions.imp.numpy import NumpyTransformation

        target_cls = NumpyTransformation

        if self._base_backend == "tensorflow":
            from py_benchmark_functions.imp.tensorflow import TensorflowTransformation

            target_cls = TensorflowTransformation
        elif self._base_backend == "torch":
            from py_benchmark_functions.imp.torch import TorchTransformation

            target_cls = TorchTransformation

        # Apply transformations
        for t in self._transformations:
            fn = target_cls(fn, **t)

        return fn

    def _select_backend(self, backend: str):
        assert backend in Registry.backends
        if self._base_fn_name is not None:
            assert (self._base_fn_name, backend) in Registry

        self._base_backend = backend
