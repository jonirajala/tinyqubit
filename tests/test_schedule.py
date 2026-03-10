"""Tests for ASAP scheduling and idle-period detection."""
from tinyqubit import Circuit, Gate, Target
from tinyqubit.passes.schedule import circuit_duration, idle_periods
from tinyqubit.passes.schedule import asap_times
from tinyqubit.hardware.devices import IBM_EAGLE_R3


# Durations matching IBM Eagle r3
DUR = {Gate.SX: 32, Gate.RZ: 0, Gate.CX: 64, Gate.MEASURE: 1120, Gate.RESET: 1120}


def test_asap_times_simple():
    """RZ(0dt) then CX(64dt) on same qubit — CX starts at 0 since RZ is 0dt."""
    c = Circuit(2)
    c.rz(0, 1.0)
    c.cx(0, 1)
    times = asap_times(c, DUR)
    assert times == [0, 0]


def test_asap_times_sequential():
    """Two CX gates sharing a qubit must be sequential."""
    c = Circuit(3)
    c.cx(0, 1)
    c.cx(1, 2)
    times = asap_times(c, DUR)
    assert times == [0, 64]


def test_circuit_duration():
    """SX(32dt) on q0, then CX(64dt) on q0,q1 — total = 32 + 64 = 96dt."""
    c = Circuit(2)
    c.sx(0)
    c.cx(0, 1)
    dur = circuit_duration(c, IBM_EAGLE_R3)
    assert dur == 96


def test_circuit_duration_no_durations():
    """Target without durations returns 0."""
    no_dur_target = Target(n_qubits=2, edges=frozenset({(0, 1)}), basis_gates=frozenset({Gate.CX}))
    c = Circuit(2)
    c.cx(0, 1)
    assert circuit_duration(c, no_dur_target) == 0


def test_circuit_duration_empty():
    assert circuit_duration(Circuit(2), IBM_EAGLE_R3) == 0


def test_idle_periods_parallel():
    """q0 gets SX(32dt)+CX(64dt)=96dt, q1 idle until CX at t=32. Gap on q1: [0,32)."""
    c = Circuit(2)
    c.sx(0)
    c.cx(0, 1)
    gaps = idle_periods(c, IBM_EAGLE_R3)
    assert (1, 0, 32) in gaps


def test_idle_periods_trailing():
    """q1 has no ops → full-circuit idle."""
    c = Circuit(2)
    c.sx(0)
    gaps = idle_periods(c, IBM_EAGLE_R3)
    # q1 idle for entire circuit duration (32dt)
    assert (1, 0, 32) in gaps


def test_idle_periods_empty():
    assert idle_periods(Circuit(2), IBM_EAGLE_R3) == []


def test_idle_periods_no_durations():
    no_dur = Target(n_qubits=2, edges=frozenset({(0, 1)}), basis_gates=frozenset({Gate.CX}))
    c = Circuit(2)
    c.cx(0, 1)
    assert idle_periods(c, no_dur) == []
