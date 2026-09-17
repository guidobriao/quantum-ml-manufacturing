from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane


def test_bell_like_circuit_on_local_aer() -> None:
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    counts = AerSimulator().run(
        circuit, shots=256, seed_simulator=2026
    ).result().get_counts()

    assert counts == {"11": 256}


def test_circuit_with_ibm_fake_backend_noise_on_local_aer() -> None:
    fake_backend = FakeBrisbane()
    simulator = AerSimulator.from_backend(fake_backend)
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    compiled = transpile(circuit, simulator, seed_transpiler=2026)

    counts = simulator.run(
        compiled, shots=256, seed_simulator=2026
    ).result().get_counts()

    assert sum(counts.values()) == 256
    assert counts.get("1", 0) > counts.get("0", 0)

