"""Hardware backend adapters: IBM Quantum, AWS Braket."""
from __future__ import annotations

from .ibm_native import submit_ibm, wait_ibm, list_ibm_backends, ibm_target, IBMJob, IBMBackend
from .braket import submit_to_braket, get_braket_results, BraketBackend
from .devices import (
    IBM_EAGLE_R3, IBM_BRISBANE, IBM_OSAKA, IBM_KYOTO,
    GOOGLE_SYCAMORE,
    IONQ_HARMONY, IONQ_ARIA, IONQ_FORTE,
    QUANTINUUM_H2, QUANTINUUM_HELIOS,
    RIGETTI_ANKAA, IQM_GARNET, IQM_SPARK,
)

__all__ = [
    "submit_ibm", "wait_ibm", "list_ibm_backends", "ibm_target", "IBMJob", "IBMBackend",
    "submit_to_braket", "get_braket_results", "BraketBackend",
    "IBM_EAGLE_R3", "IBM_BRISBANE", "IBM_OSAKA", "IBM_KYOTO",
    "GOOGLE_SYCAMORE",
    "IONQ_HARMONY", "IONQ_ARIA", "IONQ_FORTE",
    "QUANTINUUM_H2", "QUANTINUUM_HELIOS",
    "RIGETTI_ANKAA", "IQM_GARNET", "IQM_SPARK",
]
