"""Core classes for implementing new functions."""

import abc
from typing import List, NamedTuple, Union

from py_benchmark_functions.core.metadata import Metadata


class Domain(NamedTuple):
    min: List[float]
    max: List[float]


class Function:
    """Function base class."""

    def __init__(
        self,
        dims: int,
        domain_min: Union[float, List[float]] = None,
        domain_max: Union[float, List[float]] = None,
        domain: Domain = None,
    ):
        has_domain = (domain is not None) or all(
            v is not None for v in (domain_min, domain_max)
        )
        assert has_domain, "Domain must not be none."
        assert dims > 0, "Dimensions has to be greater than 0."

        # Maybe domain is not set
        if domain is None:
            domain = Domain(
                *[
                    [v] * dims if isinstance(v, float) else v
                    for v in (domain_min, domain_max)
                ]
            )

        # Assert domain and dims are compatible
        for v in domain:
            if isinstance(v, list):
                assert len(v) == dims, "Dimensions and domain must have same length."

        #  Store domain and dims
        self._domain = domain
        self._dims = dims

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    @abc.abstractmethod
    def metadata(self) -> Metadata:
        """Return the function metadata associated
        with its name.

        Returns:
            Metadata: function metadata.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the function canonical name,
        as it appears in references.

        Returns:
            str: function canonical name.
        """
        pass

    @abc.abstractmethod
    def grads(self, x):
        """Returns the gradients of the function at x."""
        pass

    @abc.abstractmethod
    def grads_at(self, x):
        """Returns a tuple containing f(x) and grad(x)."""
        pass

    @abc.abstractmethod
    def _fn(self, x):
        """
        This method must have the function implementation.
        """
        pass

    def __call__(self, x):
        return self._fn(x)

    def __str__(self):
        return f"{self.name}(domain={self.domain})"
