"""IBM Quantum backend adapter."""
from __future__ import annotations


def submit_to_ibm(circuit, backend_name: str = "ibm_brisbane", shots: int = 1024):
    """Submit circuit to IBM Quantum hardware, returns RuntimeJob."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except ImportError:
        raise ImportError("qiskit-ibm-runtime required: pip install qiskit-ibm-runtime")

    from ..qiskit_adapter import to_qiskit

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    qc = to_qiskit(circuit)
    transpiled = generate_preset_pass_manager(backend=backend, optimization_level=1).run(qc)
    return SamplerV2(backend).run([transpiled], shots=shots)


def get_ibm_results(job, timeout: float = 600) -> dict[str, int]:
    """Get measurement counts from IBM job. Returns {"00": 512, "11": 512}."""
    return job.result(timeout=timeout)[0].data.c.get_counts()
