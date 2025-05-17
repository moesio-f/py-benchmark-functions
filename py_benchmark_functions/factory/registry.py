"""Function registry."""

import importlib
import inspect
from functools import cached_property
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from py_benchmark_functions.core import Function, Transformation


class FunctionRegistry:
    def __init__(self):
        self._reg: Dict[str, Dict[str, type]] = dict()

        for m in ["numpy", "tensorflow", "torch"]:
            self._register_from_module(
                f"py_benchmark_functions.imp.{m}", m, m.capitalize()
            )

    @cached_property
    def backends(self) -> Set[str]:
        return set(v_ for k, v in self._reg.items() for v_ in v.keys())

    @cached_property
    def functions(self) -> List[str]:
        return list(sorted(self._reg))

    def get(self, fn: str, backend: Optional[str] = None) -> Type[Function]:
        if backend is None:
            backend = "numpy"

        if fn not in self._reg:
            raise KeyError(f"Function '{fn}' not in registry.")

        if backend not in self.backends:
            raise KeyError(
                f"Unknown backend '{backend}'. Only {self.backends} are available."
            )

        if backend not in self._reg[fn]:
            raise KeyError(
                f"Function '{fn}' currently doesn't has {backend} implementation."
            )

        return self._reg[fn][backend]

    def __getitem__(self, key: Union[str, Tuple[str, str]]) -> Type[Function]:
        fn, backend = key, None
        if isinstance(key, tuple):
            fn, backend = key

        return self.get(fn, backend)

    def __contains__(self, key: Union[str, Tuple[str, str]]):
        fn, backend = key, None
        if isinstance(key, tuple):
            fn, backend = key

        if fn not in self._reg:
            return False

        if backend is not None:
            return backend in self._reg[fn]

        return True

    def __len__(self) -> int:
        return len(self._reg)

    def __iter__(self) -> str:
        for k in self._reg:
            yield k

    def _register_from_module(self, module_name: str, imp_backend: str, suffix: str):
        try:
            functions = {
                obj.__name__.replace(suffix, ""): obj
                for _, obj in inspect.getmembers(
                    importlib.import_module(module_name),
                    predicate=lambda o: self._is_obj_function_class(o, suffix),
                )
            }

            for k, v in functions.items():
                if k not in self._reg:
                    self._reg[k] = dict()

                assert imp_backend not in self._reg[k]
                self._reg[k][imp_backend] = v
        except ImportError:
            pass

    def _is_obj_function_class(self, obj, suffix) -> bool:
        if not inspect.isclass(obj):
            return False

        if not (issubclass(obj, Function) and not issubclass(obj, Transformation)):
            return False

        if not obj.__name__.endswith(suffix):
            return False

        return True


Registry = FunctionRegistry()
