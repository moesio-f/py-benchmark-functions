from py_benchmark_functions import core


class FunctionEvaluationExamples:
    default_str = "default"

    default_x_4d: dict[str, list[float]] = {
        "Mishra2": [0.0, 0.25, 0.5, 0.75],
        default_str: [1.0, 2.0, 3.0, 4.0],
    }
    default_fx_4d: dict[str, float] = {
        "Ackley": 8.43469444443746497,
        "Alpine2": -0.40033344730936005,
        "BentCigar": 29000001.0,
        "Brown": 1.6678281e16,
        "ChungReynolds": 900.0,
        "Csendes": 11063.416256526398,
        "Deb1": -6.182844847431069e-87,
        "Deb3": -0.036599504738713866,
        "DixonPrice": 4230.0,
        "Griewank": 1.00187037800320189,
        "Levy": 2.76397190019909811,
        "Mishra2": 49.12257870688604,
        "PowellSum": 1114.0,
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
        "StrechedVSineWave": 3.4648263408031923,
        "SumSquares": 100.0,
        "Trigonometric2": 42.98949432373047,
        "WWavy": 1.1130512151573806,
        "Weierstrass": 23.999988555908203,
        "Whitley": 367532.562701215,
        "Zakharov": 50880.0,
    }

    defaults_x: dict[int, dict[str, list[float]]] = {
        4: default_x_4d,
    }
    defaults_fx: dict[int, dict[str, float]] = {
        4: default_fx_4d,
    }

    zeros_x_4d: list[float] = [0.0, 0.0, 0.0, 0.0]
    zeros_fx_4d: dict[str, float] = {
        "Ackley": 4.44089209850062616e-16,
        "Alpine2": 0.0,
        "BentCigar": 0.0,
        "Brown": 0.0,
        "ChungReynolds": 0.0,
        "Csendes": 0.0,
        "Deb1": 0.0,
        "Deb3": -0.1249999850988388,
        "DixonPrice": 1.0,
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
        "Weierstrass": 23.999988555908203,
        "Whitley": 7.359163106109763,
        "Zakharov": 0.0,
    }

    zeros_x: dict[int, list[float]] = {
        4: zeros_x_4d,
    }
    zeros_fx: dict[int, dict[str, float]] = {
        4: zeros_fx_4d,
    }

    @classmethod
    def get_default_eval(cls, function: core.Function, dims: int) -> tuple[list, float]:
        assert dims in cls.defaults_x and dims in cls.defaults_fx
        fn_str = function.name
        x_dict = cls.defaults_x[dims]
        fx_dict = cls.defaults_fx[dims]

        query_arr = x_dict[fn_str] if fn_str in x_dict else x_dict[cls.default_str]
        result = fx_dict[fn_str]

        return query_arr, result

    @classmethod
    def get_eval_at_zeros(
        cls, function: core.Function, dims: int
    ) -> tuple[list, float]:
        assert dims in cls.zeros_x and dims in cls.zeros_fx
        fx_dict = cls.zeros_fx[dims]

        query_arr = cls.zeros_x[dims]
        result = fx_dict[function.name]

        return query_arr, result
