"""Diagnose the thesis environment without accessing IBM Quantum services."""

from __future__ import annotations

from importlib import import_module, metadata
import os
from pathlib import Path
import platform
import sys

# Keep generated plotting/font caches local and outside version control.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "results" / ".cache"))

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane


PACKAGES = {
    "qiskit": "qiskit",
    "qiskit-aer": "qiskit_aer",
    "qiskit-machine-learning": "qiskit_machine_learning",
    "qiskit-ibm-runtime": "qiskit_ibm_runtime",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "jupyter": "jupyter",
    "pytest": "pytest",
}


def aer_check() -> dict[str, object]:
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    simulator = AerSimulator()
    counts = simulator.run(circuit, shots=256, seed_simulator=2026).result().get_counts()
    return {"backend": simulator.name, "counts": counts, "passed": counts == {"11": 256}}


def fake_backend_check() -> dict[str, object]:
    fake_backend = FakeBrisbane()
    simulator = AerSimulator.from_backend(fake_backend)
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    compiled = transpile(circuit, simulator, seed_transpiler=2026)
    counts = simulator.run(compiled, shots=256, seed_simulator=2026).result().get_counts()
    passed = sum(counts.values()) == 256 and counts.get("1", 0) > counts.get("0", 0)
    return {
        "fake_backend": fake_backend.name,
        "local_simulator": simulator.name,
        "counts": counts,
        "passed": passed,
    }


def main() -> int:
    expected_interpreter = PROJECT_ROOT / ".venv" / "bin" / "python"

    print(f"Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}")
    print(f"Expected local interpreter: {expected_interpreter}")
    print(f"Using project .venv: {Path(sys.executable).resolve() == expected_interpreter.resolve()}")

    imports_ok = True
    print("\nInstalled packages and imports:")
    for distribution, module in PACKAGES.items():
        try:
            import_module(module)
            version = metadata.version(distribution)
            print(f"  PASS {distribution}=={version} ({module})")
        except Exception as exc:  # pragma: no cover - diagnostic reporting
            imports_ok = False
            print(f"  FAIL {distribution} ({module}): {type(exc).__name__}: {exc}")

    aer = aer_check()
    fake = fake_backend_check()
    print(f"\nAer circuit: {aer}")
    print(f"IBM fake backend (local Aer simulation): {fake}")

    return 0 if imports_ok and aer["passed"] and fake["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
