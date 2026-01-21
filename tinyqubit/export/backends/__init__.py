"""Hardware backend adapters: IBM, AWS Braket."""
from .ibm import submit_to_ibm, get_ibm_results
from .braket import submit_to_braket, get_braket_results

__all__ = ["submit_to_ibm", "get_ibm_results", "submit_to_braket", "get_braket_results"]
