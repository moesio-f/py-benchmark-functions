# Library configuration
import os

PREFIX: str = "PY_BENCHMARK_FUNCTIONS"

CHECK_INPUT_SHAPE: bool = bool(int(os.getenv(f"{PREFIX}_CHECK_INPUT_SHAPE", "1")))
CHECK_INPUT_DOMAIN: bool = bool(int(os.getenv(f"{PREFIX}_CHECK_INPUT_DOMAIN", "1")))
COERCE_INPUT_TO_DOMAIN: bool = bool(
    int(os.getenv(f"{PREFIX}_CHECK_INPUT_TO_DOMAIN", "1"))
)

del PREFIX, os
