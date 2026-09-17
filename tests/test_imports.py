from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "qiskit",
        "qiskit_aer",
        "qiskit_machine_learning",
        "qiskit_ibm_runtime",
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "matplotlib",
        "seaborn",
        "jupyter",
    ],
)
def test_required_module_imports(module: str) -> None:
    assert import_module(module) is not None

