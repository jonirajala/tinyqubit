"""Reference hardware topologies for testing and offline compilation.

NOTE: These are static snapshots. For live IBM backends, use ibm_target().
"""
from __future__ import annotations

from ..target import Target
from ..ir import Gate

_IBM_EAGLE_BASIS = frozenset({Gate.SX, Gate.RZ, Gate.CX})
_IBM_HERON_BASIS = frozenset({Gate.X, Gate.SX, Gate.RZ, Gate.CZ})
_IONQ_BASIS = frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.CX})
_QUANTINUUM_BASIS = frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.RZZ})
_CZ_BASIS = frozenset({Gate.RX, Gate.RZ, Gate.CZ})

def _all_to_all(n):
    return frozenset((i, j) for i in range(n) for j in range(i + 1, n))

# IBM Eagle r3 heavy-hex coupling map (127Q, directed CX)
# Source: qiskit-ibm-runtime FakeBrisbane conf_brisbane.json
_IBM_EAGLE_EDGES = frozenset({
    (1,0), (2,1), (3,2), (4,3), (4,5), (4,15), (6,5), (6,7),
    (7,8), (8,9), (10,9), (10,11), (11,12), (12,17), (13,12), (14,0),
    (14,18), (15,22), (16,8), (16,26), (17,30), (18,19), (20,19), (20,33),
    (21,20), (21,22), (22,23), (24,23), (24,34), (25,24), (26,25), (27,26),
    (28,27), (28,29), (28,35), (30,29), (30,31), (31,32), (32,36), (33,39),
    (34,43), (35,47), (36,51), (37,38), (39,38), (40,39), (40,41), (41,53),
    (42,41), (42,43), (43,44), (44,45), (46,45), (46,47), (48,47), (48,49),
    (50,49), (50,51), (52,37), (52,56), (53,60), (54,45), (54,64), (55,49),
    (55,68), (56,57), (57,58), (58,59), (58,71), (59,60), (60,61), (62,61),
    (62,63), (62,72), (63,64), (65,64), (65,66), (67,66), (67,68), (69,68),
    (69,70), (73,66), (74,70), (74,89), (75,90), (76,75), (77,71), (77,76),
    (77,78), (79,78), (79,80), (80,81), (81,72), (81,82), (82,83), (83,92),
    (84,83), (85,73), (85,84), (85,86), (86,87), (87,88), (88,89), (91,79),
    (92,102), (93,87), (93,106), (94,90), (94,95), (95,96), (97,96), (97,98),
    (98,91), (99,98), (100,99), (100,110), (101,100), (101,102), (102,103), (104,103),
    (105,104), (105,106), (107,106), (108,107), (108,112), (109,96), (110,118), (111,104),
    (112,126), (113,114), (114,109), (114,115), (116,115), (116,117), (117,118), (118,119),
    (120,119), (121,120), (122,111), (122,121), (122,123), (124,123), (125,124), (125,126),
})

# Google Sycamore diamond grid (54Q, undirected CZ)
# Source: cirq_google.Sycamore, qubits linearized row-major from (0,5) to (9,4)
_GOOGLE_SYCAMORE_EDGES = frozenset({
    (0,1), (0,3), (1,4), (2,3), (2,7), (3,4), (3,8), (4,5), (4,9), (5,10),
    (6,7), (6,13), (7,8), (7,14), (8,9), (8,15), (9,10), (9,16), (10,11), (10,17),
    (11,18), (12,13), (12,21), (13,14), (13,22), (14,15), (14,23), (15,16), (15,24),
    (16,17), (16,25), (17,18), (17,26), (18,19), (18,27), (19,28), (20,21), (20,30),
    (21,22), (21,31), (22,23), (22,32), (23,24), (23,33), (24,25), (24,34), (25,26),
    (25,35), (26,27), (26,36), (27,28), (27,37), (29,30), (30,31), (30,38), (31,32),
    (31,39), (32,33), (32,40), (33,34), (33,41), (34,35), (34,42), (35,36), (35,43),
    (36,37), (36,44), (38,39), (39,40), (39,45), (40,41), (40,46), (41,42), (41,47),
    (42,43), (42,48), (43,44), (43,49), (45,46), (46,47), (46,50), (47,48), (47,51),
    (48,49), (48,52), (50,51), (51,52), (51,53),
})

# Rigetti Ankaa-2 octagonal lattice (84Q, undirected CZ)
# NOTE: qubits 42 and 48 are absent from the topology
_RIGETTI_ANKAA_EDGES = frozenset({
    (0,1), (0,7), (1,2), (1,8), (2,3), (2,9), (3,4), (3,10),
    (4,5), (4,11), (5,6), (5,12), (6,13), (7,8), (7,14), (8,9),
    (8,15), (9,10), (9,16), (10,11), (11,12), (11,18), (12,13), (12,19),
    (13,20), (14,15), (14,21), (15,16), (15,22), (16,17), (16,23), (17,18),
    (17,24), (18,19), (18,25), (19,20), (19,26), (20,27), (21,22), (21,28),
    (22,23), (22,29), (23,24), (23,30), (24,25), (24,31), (25,26), (25,32),
    (26,27), (26,33), (28,29), (28,35), (29,30), (29,36), (30,31), (30,37),
    (31,38), (32,33), (32,39), (33,34), (33,40), (34,41), (35,36), (36,37),
    (36,43), (37,38), (37,44), (38,39), (38,45), (39,40), (39,46), (40,41),
    (40,47), (43,50), (44,45), (44,51), (45,46), (45,52), (46,47), (46,53),
    (47,54), (49,50), (49,56), (50,51), (50,57), (51,52), (51,58), (52,53),
    (52,59), (53,54), (53,60), (54,55), (54,61), (55,62), (56,57), (56,63),
    (57,58), (57,64), (58,59), (58,65), (59,60), (59,66), (60,61), (60,67),
    (61,62), (61,68), (62,69), (63,64), (63,70), (64,65), (64,71), (65,66),
    (65,72), (66,73), (67,68), (67,74), (68,69), (68,75), (69,76), (70,71),
    (70,77), (71,72), (71,78), (72,73), (72,79), (73,74), (73,80), (74,75),
    (74,81), (75,76), (75,82), (76,83), (77,78), (78,79), (79,80), (80,81),
    (81,82), (82,83),
})

# IQM Garnet square lattice (20Q, undirected CZ)
# Source: qiskit-on-iqm fake_garnet.py
_IQM_GARNET_EDGES = frozenset({
    (0,1), (0,3), (1,4), (2,3), (2,7), (3,4), (3,8), (4,5),
    (4,9), (5,6), (5,10), (6,11), (7,8), (7,12), (8,9), (8,13),
    (9,10), (9,14), (10,11), (10,15), (11,16), (12,13), (13,14),
    (13,17), (14,15), (14,18), (15,16), (15,19), (17,18), (18,19),
})

# IQM Spark star topology (5Q, undirected CZ)
_IQM_SPARK_EDGES = frozenset({(0,2), (1,2), (2,3), (2,4)})


# --- IBM ---
IBM_EAGLE_R3 = Target(n_qubits=127, edges=_IBM_EAGLE_EDGES, basis_gates=_IBM_EAGLE_BASIS, name="ibm_eagle_r3", directed=True,
                       duration={Gate.SX: 32, Gate.RZ: 0, Gate.CX: 64, Gate.MEASURE: 1120, Gate.RESET: 1120})
IBM_BRISBANE = IBM_EAGLE_R3  # retired 2025-11-03
IBM_OSAKA = IBM_EAGLE_R3     # retired 2024-08-13
IBM_KYOTO = IBM_EAGLE_R3     # retired 2024-09-05

# --- Google ---
GOOGLE_SYCAMORE = Target(n_qubits=54, edges=_GOOGLE_SYCAMORE_EDGES, basis_gates=_CZ_BASIS, name="google_sycamore")

# --- IonQ (trapped ion, all-to-all) ---
IONQ_HARMONY = Target(n_qubits=11, edges=_all_to_all(11), basis_gates=_IONQ_BASIS, name="ionq_harmony")  # retired 2024-09
IONQ_ARIA = Target(n_qubits=25, edges=_all_to_all(25), basis_gates=_IONQ_BASIS, name="ionq_aria")
IONQ_FORTE = Target(n_qubits=36, edges=_all_to_all(36), basis_gates=_IONQ_BASIS, name="ionq_forte")

# --- Quantinuum (trapped ion, all-to-all, RZZ-native) ---
QUANTINUUM_H2 = Target(n_qubits=56, edges=_all_to_all(56), basis_gates=_QUANTINUUM_BASIS, name="quantinuum_h2")
QUANTINUUM_HELIOS = Target(n_qubits=98, edges=_all_to_all(98), basis_gates=_QUANTINUUM_BASIS, name="quantinuum_helios")

# --- Rigetti (superconducting, CZ-native) ---
RIGETTI_ANKAA = Target(n_qubits=84, edges=_RIGETTI_ANKAA_EDGES, basis_gates=_CZ_BASIS, name="rigetti_ankaa")

# --- IQM (superconducting, CZ-native) ---
IQM_GARNET = Target(n_qubits=20, edges=_IQM_GARNET_EDGES, basis_gates=_CZ_BASIS, name="iqm_garnet")
IQM_SPARK = Target(n_qubits=5, edges=_IQM_SPARK_EDGES, basis_gates=_CZ_BASIS, name="iqm_spark")
