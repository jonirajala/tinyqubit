"""IBM Quantum native REST client — zero vendor SDK, stdlib only."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_IBM_CLOUD = "https://quantum.cloud.ibm.com/api/v1"
_IAM_URL = "https://iam.cloud.ibm.com/identity/token"
_API_VERSION = "2025-01-01"
_USER_AGENT = "tinyqubit/1.0"


def _resolve_creds(api_key: str | None, crn: str | None) -> tuple[str, str | None]:
    api_key = api_key or os.environ.get("IBM_API_KEY", "")
    if not api_key:
        raise ValueError("api_key required — pass directly or set IBM_API_KEY env var")
    return api_key, crn or os.environ.get("IBM_CRN") or None


@dataclass
class IBMJob:
    job_id: str
    _auth: _IBMAuth
    backend: str


class _IBMAuth:
    def __init__(self, api_key: str, crn: str | None = None):
        self.api_key, self.crn = api_key, crn
        self._token: str | None = None
        self._expiration: float = 0

    def _refresh(self):
        body = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={self.api_key}".encode()
        req = Request(_IAM_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        resp = json.loads(urlopen(req).read())
        self._token, self._expiration = resp["access_token"], resp["expiration"]

    def _discover_crn(self):
        if self._token is None or time.time() >= self._expiration - 60:
            self._refresh()
        h = {"Authorization": f"Bearer {self._token}", "IBM-API-Version": _API_VERSION,
             "Accept": "application/json", "User-Agent": _USER_AGENT}
        try:
            resp = json.loads(urlopen(Request(f"{_IBM_CLOUD}/instances", headers=h)).read())
            instances = resp if isinstance(resp, list) else resp.get("instances", [])
            if not instances:
                raise RuntimeError("No instances found")
            self.crn = instances[0]["crn"]
        except Exception as e:
            raise RuntimeError(f"CRN auto-discovery failed ({e}). Pass crn= or set IBM_CRN env var.") from e

    def headers(self) -> dict[str, str]:
        if self._token is None or time.time() >= self._expiration - 60:
            self._refresh()
        if self.crn is None:
            self._discover_crn()
        return {"Authorization": f"Bearer {self._token}", "Service-CRN": self.crn,
                "IBM-API-Version": _API_VERSION, "Content-Type": "application/json", "User-Agent": _USER_AGENT}


def list_ibm_backends(api_key: str | None = None, crn: str | None = None) -> list[dict]:
    """List available IBM Quantum backends. Returns [{name, n_qubits, basis_gates, status}]."""
    api_key, crn = _resolve_creds(api_key, crn)
    resp = json.loads(urlopen(Request(f"{_IBM_CLOUD}/backends", headers=_IBMAuth(api_key, crn).headers())).read())
    backends = resp if isinstance(resp, list) else resp.get("backends", resp.get("devices", []))
    return [{"name": b["name"], "n_qubits": b.get("num_qubits", b.get("n_qubits", "?")),
             "basis_gates": b.get("basis_gates", []),
             "status": s.get("name", str(s)) if isinstance(s := b.get("status", "?"), dict) else str(s)}
            for b in backends]


def ibm_target(backend_name: str, api_key: str | None = None, crn: str | None = None):
    """Query IBM backend config and return a tinyqubit Target with real coupling map."""
    from ...ir import Gate
    from ...target import Target
    api_key, crn = _resolve_creds(api_key, crn)
    req = Request(f"{_IBM_CLOUD}/backends/{backend_name}/configuration", headers=_IBMAuth(api_key, crn).headers())
    try:
        config = json.loads(urlopen(req).read())
    except HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Failed to get config for {backend_name} (HTTP {e.code}): {body}") from e
    basis = frozenset(Gate[g.upper()] for g in config.get("basis_gates", []) if g.upper() in Gate.__members__)
    edges = frozenset(tuple(pair) for pair in config.get("coupling_map", []))
    return Target(n_qubits=config["n_qubits"], edges=edges, basis_gates=basis, name=backend_name, directed=True)


def submit_ibm(circuit, backend: str = "ibm_brisbane", shots: int = 4096,
               api_key: str | None = None, crn: str | None = None, target=None) -> IBMJob:
    """Submit a compiled circuit to IBM Quantum via REST. Returns IBMJob."""
    api_key, crn = _resolve_creds(api_key, crn)
    if target is not None:
        from ...target import validate
        errors = validate(circuit, target)
        if errors:
            raise ValueError("Circuit validation failed:\n" + "\n".join(errors))

    from ..qasm import to_openqasm3
    qasm = to_openqasm3(circuit, include_mapping=False, physical_qubits=True)
    auth = _IBMAuth(api_key, crn)
    payload = json.dumps({"program_id": "sampler", "backend": backend,
                          "params": {"pubs": [[qasm, None, shots]], "version": 2}}).encode()
    try:
        resp = json.loads(urlopen(Request(f"{_IBM_CLOUD}/jobs", data=payload, headers=auth.headers())).read())
    except HTTPError as e:
        raise RuntimeError(f"IBM job submission failed (HTTP {e.code}): {e.read().decode(errors='replace')}") from e
    return IBMJob(job_id=resp["id"], _auth=auth, backend=backend)


def wait_ibm(job: IBMJob, timeout: float = 600, poll_interval: float = 2) -> dict[str, int]:
    """Poll until job completes, return measurement counts dict."""
    deadline, interval = time.time() + timeout, poll_interval
    while True:
        resp = json.loads(urlopen(Request(f"{_IBM_CLOUD}/jobs/{job.job_id}", headers=job._auth.headers())).read())
        status = resp["status"]
        if status == "Completed":
            break
        if status == "Failed":
            raise RuntimeError(f"IBM job {job.job_id} failed: {resp.get('reason') or json.dumps(resp, indent=2)}")
        if time.time() > deadline:
            raise TimeoutError(f"IBM job {job.job_id} timed out after {timeout}s (status: {status})")
        time.sleep(interval)
        interval = min(interval * 1.5, 30)

    req = Request(f"{_IBM_CLOUD}/jobs/{job.job_id}/results", headers=job._auth.headers())
    try:
        results = json.loads(urlopen(req).read())
    except HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Failed to fetch results for {job.job_id} (HTTP {e.code}): {body}") from e

    # NOTE: hex samples → counts, zero-padded to inferred bit width
    pub = results["results"][0]
    counts: dict[str, int] = {}
    if "data" in pub and "c" in pub["data"]:
        samples = pub["data"]["c"]["samples"]
        n_bits = len(samples[0].replace("0x", "")) * 4 if samples else 0
        for hex_val in samples:
            bits = bin(int(hex_val, 16))[2:].zfill(n_bits)
            counts[bits] = counts.get(bits, 0) + 1
    return counts
