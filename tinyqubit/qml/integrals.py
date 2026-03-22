"""STO-3G Gaussian integral engine: basis data, McMurchie-Davidson integrals, RHF-SCF, CASCI."""
from __future__ import annotations
import numpy as np
from math import pi as _pi, exp as _exp

# STO-3G basis parameters (H through Ne) -------
# Each entry: list of (angular_momentum, exponents, coefficients)
# angular_momentum: 0=s, 1=p
_STO3G: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {
    1: [(0, np.array([3.42525091, 0.62391373, 0.16885540]),
            np.array([0.15432897, 0.53532814, 0.44463454]))],
    2: [(0, np.array([6.36242139, 1.15892300, 0.31364979]),
            np.array([0.15432897, 0.53532814, 0.44463454]))],
    3: [(0, np.array([16.1195750, 2.93620070, 0.79465370]),
            np.array([0.15432897, 0.53532814, 0.44463454])),
        (0, np.array([0.63628970, 0.14786010, 0.04808870]),
            np.array([-0.09996723, 0.39951283, 0.70011547])),
        (1, np.array([0.63628970, 0.14786010, 0.04808870]),
            np.array([0.15591627, 0.60768372, 0.39195739]))],
    4: [(0, np.array([30.1678710, 5.49532230, 1.48719270]),
            np.array([0.15432897, 0.53532814, 0.44463454])),
        (0, np.array([1.31483320, 0.30553890, 0.09937070]),
            np.array([-0.09996723, 0.39951283, 0.70011547])),
        (1, np.array([1.31483320, 0.30553890, 0.09937070]),
            np.array([0.15591627, 0.60768372, 0.39195739]))],
    5: [(0, np.array([48.7911130, 8.88736220, 2.40598820]),
            np.array([0.15432897, 0.53532814, 0.44463454])),
        (0, np.array([2.23695610, 0.51982050, 0.16906180]),
            np.array([-0.09996723, 0.39951283, 0.70011547])),
        (1, np.array([2.23695610, 0.51982050, 0.16906180]),
            np.array([0.15591627, 0.60768372, 0.39195739]))],
    6: [(0, np.array([71.6168370, 13.0450960, 3.53051220]),
            np.array([0.15432897, 0.53532814, 0.44463454])),
        (0, np.array([2.94124940, 0.68348310, 0.22228990]),
            np.array([-0.09996723, 0.39951283, 0.70011547])),
        (1, np.array([2.94124940, 0.68348310, 0.22228990]),
            np.array([0.15591627, 0.60768372, 0.39195739]))],
    7: [(0, np.array([99.1061690, 18.0523120, 4.88566020]),
            np.array([0.15432897, 0.53532814, 0.44463454])),
        (0, np.array([3.78045590, 0.87849660, 0.28571440]),
            np.array([-0.09996723, 0.39951283, 0.70011547])),
        (1, np.array([3.78045590, 0.87849660, 0.28571440]),
            np.array([0.15591627, 0.60768372, 0.39195739]))],
    8: [(0, np.array([130.709320, 23.8088610, 6.44360830]),
            np.array([0.15432897, 0.53532814, 0.44463454])),
        (0, np.array([5.03315130, 1.16959610, 0.38038900]),
            np.array([-0.09996723, 0.39951283, 0.70011547])),
        (1, np.array([5.03315130, 1.16959610, 0.38038900]),
            np.array([0.15591627, 0.60768372, 0.39195739]))],
    9: [(0, np.array([166.679130, 30.3608120, 8.21682070]),
            np.array([0.15432897, 0.53532814, 0.44463454])),
        (0, np.array([6.46480320, 1.50228120, 0.48858850]),
            np.array([-0.09996723, 0.39951283, 0.70011547])),
        (1, np.array([6.46480320, 1.50228120, 0.48858850]),
            np.array([0.15591627, 0.60768372, 0.39195739]))],
    10: [(0, np.array([207.015600, 37.7081510, 10.2052970]),
             np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([8.24631510, 1.91615840, 0.62322930]),
             np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([8.24631510, 1.91615840, 0.62322930]),
             np.array([0.15591627, 0.60768372, 0.39195739]))],
}

_631G: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {
    1: [(0, np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.03349460, 0.23472695, 0.81375733])),
        (0, np.array([0.1612778]), np.array([1.0]))],
    2: [(0, np.array([38.4216340, 5.7780300, 1.2417740]), np.array([0.04013974, 0.26124610, 0.79318462])),
        (0, np.array([0.2979640]), np.array([1.0]))],
    3: [(0, np.array([642.418915, 96.798515, 22.091121, 6.201070, 1.935118, 0.636736]),
            np.array([0.002143, 0.016209, 0.077316, 0.245786, 0.470189, 0.345471])),
        (0, np.array([2.324918, 0.632430, 0.079053]), np.array([-0.035092, -0.191233, 1.083988])),
        (1, np.array([2.324918, 0.632430, 0.079053]), np.array([0.008942, 0.141009, 0.945364])),
        (0, np.array([0.035962]), np.array([1.0])),
        (1, np.array([0.035962]), np.array([1.0]))],
    4: [(0, np.array([1264.58569, 189.93681, 43.159089, 12.098663, 3.806323, 1.272890]),
            np.array([0.001945, 0.014835, 0.072091, 0.237154, 0.469199, 0.356520])),
        (0, np.array([3.196463, 0.747813, 0.219966]), np.array([-0.112649, -0.229506, 1.186917])),
        (1, np.array([3.196463, 0.747813, 0.219966]), np.array([0.055980, 0.261551, 0.793972])),
        (0, np.array([0.082310]), np.array([1.0])),
        (1, np.array([0.082310]), np.array([1.0]))],
    5: [(0, np.array([2068.88225, 310.64957, 70.683033, 19.861080, 6.299305, 2.127027]),
            np.array([0.001866, 0.014251, 0.069552, 0.232573, 0.467079, 0.363431])),
        (0, np.array([4.727971, 1.190338, 0.359412]), np.array([-0.130394, -0.130789, 1.130944])),
        (1, np.array([4.727971, 1.190338, 0.359412]), np.array([0.074598, 0.307847, 0.743457])),
        (0, np.array([0.126751]), np.array([1.0])),
        (1, np.array([0.126751]), np.array([1.0]))],
    6: [(0, np.array([3047.52488, 457.36952, 103.94869, 29.210155, 9.286663, 3.163927]),
            np.array([0.001835, 0.014037, 0.068843, 0.232184, 0.467941, 0.362312])),
        (0, np.array([7.868272, 1.881289, 0.544249]), np.array([-0.119332, -0.160854, 1.143456])),
        (1, np.array([7.868272, 1.881289, 0.544249]), np.array([0.068999, 0.316424, 0.744308])),
        (0, np.array([0.168714]), np.array([1.0])),
        (1, np.array([0.168714]), np.array([1.0]))],
    7: [(0, np.array([4173.51146, 627.45791, 142.90209, 40.234329, 12.820213, 4.390437]),
            np.array([0.001835, 0.013995, 0.068587, 0.232241, 0.469070, 0.360455])),
        (0, np.array([11.626362, 2.716280, 0.772218]), np.array([-0.114961, -0.169117, 1.145852])),
        (1, np.array([11.626362, 2.716280, 0.772218]), np.array([0.067580, 0.323907, 0.740895])),
        (0, np.array([0.212031]), np.array([1.0])),
        (1, np.array([0.212031]), np.array([1.0]))],
    8: [(0, np.array([5484.67166, 825.23495, 188.04696, 52.964500, 16.897570, 5.799635]),
            np.array([0.001831, 0.013950, 0.068445, 0.232714, 0.470193, 0.358521])),
        (0, np.array([15.539616, 3.599934, 1.013762]), np.array([-0.110778, -0.148026, 1.130767])),
        (1, np.array([15.539616, 3.599934, 1.013762]), np.array([0.070874, 0.339753, 0.727159])),
        (0, np.array([0.270006]), np.array([1.0])),
        (1, np.array([0.270006]), np.array([1.0]))],
    9: [(0, np.array([7001.71309, 1051.36609, 239.28569, 67.397445, 21.519957, 7.403101]),
            np.array([0.001820, 0.013916, 0.068405, 0.233186, 0.471267, 0.356619])),
        (0, np.array([20.847953, 4.808308, 1.344070]), np.array([-0.108507, -0.146452, 1.128689])),
        (1, np.array([20.847953, 4.808308, 1.344070]), np.array([0.071629, 0.345912, 0.722470])),
        (0, np.array([0.358151]), np.array([1.0])),
        (1, np.array([0.358151]), np.array([1.0]))],
    10: [(0, np.array([8425.85153, 1268.51940, 289.62141, 81.859004, 26.251508, 9.094721]),
             np.array([0.001884, 0.014337, 0.070110, 0.237373, 0.473007, 0.348401])),
         (0, np.array([26.532131, 6.101755, 1.696272]), np.array([-0.107118, -0.146164, 1.127774])),
         (1, np.array([26.532131, 6.101755, 1.696272]), np.array([0.071910, 0.349513, 0.719941])),
         (0, np.array([0.445819]), np.array([1.0])),
         (1, np.array([0.445819]), np.array([1.0]))],
}

def _631g_row3(e1s, c1s, e_sp1, cs1, cp1, e_sp2, cs2, cp2, e_sp3):
    """Build 6-31G shells for Na-Ar (S + 3 SP shells)."""
    e1s, e_sp1, e_sp2 = np.array(e1s), np.array(e_sp1), np.array(e_sp2)
    e_sp3 = np.array([e_sp3])
    return [(0, e1s, np.array(c1s)),
            (0, e_sp1, np.array(cs1)), (1, e_sp1, np.array(cp1)),
            (0, e_sp2, np.array(cs2)), (1, e_sp2, np.array(cp2)),
            (0, e_sp3, np.array([1.0])), (1, e_sp3, np.array([1.0]))]

_631G.update({
    11: _631g_row3([9993.2,1499.89,341.951,94.6796,29.7345,10.0063],
                   [0.001938,0.014807,0.072705,0.252629,0.493242,0.313169],
                   [150.963,35.5878,11.1683,3.90201,1.38177,0.466382],
                   [-0.003542,-0.043959,-0.109752,0.187398,0.646700,0.306058],
                   [0.005002,0.035511,0.142825,0.338620,0.451579,0.273271],
                   [0.497966,0.084353,0.066635], [-0.248503,-0.131704,1.233521],
                   [-0.023023,0.950359,0.059858], 0.025954),
    12: _631g_row3([11722.8,1759.93,400.846,112.807,35.9997,12.1828],
                   [0.001978,0.015114,0.073911,0.249191,0.487928,0.319662],
                   [189.180,45.2119,14.3563,5.13886,1.90652,0.705887],
                   [-0.003237,-0.041008,-0.112600,0.148633,0.616497,0.364829],
                   [0.004928,0.034989,0.140725,0.333642,0.444940,0.269254],
                   [0.929340,0.269035,0.117379], [-0.212291,-0.107985,1.175845],
                   [-0.022419,0.192271,0.846180], 0.042106),
    13: _631g_row3([13983.1,2098.75,477.705,134.360,42.8709,14.5189],
                   [0.001943,0.014860,0.072849,0.246830,0.487258,0.323496],
                   [239.668,57.4419,18.2859,6.59914,2.49049,0.944545],
                   [-0.002926,-0.037408,-0.114487,0.115635,0.612595,0.393799],
                   [0.004603,0.033199,0.136282,0.330476,0.449146,0.265704],
                   [1.27790,0.397590,0.160095], [-0.227607,0.001446,1.092794],
                   [-0.017513,0.244533,0.804934], 0.055658),
    14: _631g_row3([16115.9,2425.58,553.867,156.340,50.0683,17.0178],
                   [0.001959,0.014929,0.072848,0.246130,0.485914,0.325002],
                   [292.718,69.8731,22.3363,8.15039,3.13458,1.22543],
                   [-0.002781,-0.035715,-0.114985,0.093563,0.603017,0.418959],
                   [0.004438,0.032668,0.134721,0.328678,0.449640,0.261372],
                   [1.72738,0.572922,0.222192], [-0.244631,0.004316,1.098185],
                   [-0.017795,0.253539,0.800669], 0.077837),
    15: _631g_row3([19413.3,2909.42,661.364,185.759,59.1943,20.0310],
                   [0.001852,0.014206,0.069999,0.240079,0.484762,0.335200],
                   [339.478,81.0101,25.8780,9.45221,3.66566,1.46746],
                   [-0.002782,-0.036050,-0.116631,0.096833,0.614418,0.403798],
                   [0.004565,0.033694,0.139755,0.339362,0.450921,0.238586],
                   [2.15623,0.748997,0.283145], [-0.252924,0.032852,1.081255],
                   [-0.017765,0.274058,0.785422], 0.099832),
    16: _631g_row3([21917.1,3301.49,754.146,212.711,67.9896,23.0515],
                   [0.001869,0.014230,0.069696,0.238487,0.483307,0.338074],
                   [423.735,100.710,32.1599,11.8079,4.63110,1.87025],
                   [-0.002377,-0.031693,-0.113317,0.056090,0.592255,0.455006],
                   [0.004061,0.030681,0.130452,0.327205,0.452851,0.256042],
                   [2.61584,0.922167,0.341287], [-0.250373,0.066957,1.054506],
                   [-0.014510,0.310263,0.754482], 0.117167),
    17: _631g_row3([25180.1,3780.35,860.474,242.145,77.3349,26.2470],
                   [0.001833,0.014034,0.069097,0.237452,0.483034,0.339856],
                   [491.765,116.984,37.4153,13.7834,5.45215,2.22588],
                   [-0.002297,-0.030714,-0.112528,0.045016,0.589353,0.465206],
                   [0.003989,0.030318,0.129880,0.327951,0.453527,0.252154],
                   [3.18649,1.14427,0.420377], [-0.251828,0.061589,1.060184],
                   [-0.014299,0.323572,0.743508], 0.142657),
    18: _631g_row3([28348.3,4257.62,969.857,273.263,87.3695,29.6867],
                   [0.001825,0.013969,0.068707,0.236204,0.482214,0.342043],
                   [575.891,136.816,43.8098,16.2094,6.46084,2.65114],
                   [-0.002160,-0.029078,-0.110827,0.027699,0.577613,0.488688],
                   [0.003807,0.029230,0.126467,0.323510,0.454896,0.256630],
                   [3.86028,1.41373,0.516646], [-0.255593,0.037807,1.080564],
                   [-0.015920,0.324646,0.743990], 0.173888),
})

_STO3G.update({
    11: [(0, np.array([250.772430, 45.678511, 12.362388]), np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([12.040193, 2.797882, 0.909958]), np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([12.040193, 2.797882, 0.909958]), np.array([0.15591627, 0.60768372, 0.39195739])),
         (0, np.array([1.478741, 0.412565, 0.161475]), np.array([-0.21962037, 0.22559543, 0.90039843])),
         (1, np.array([1.478741, 0.412565, 0.161475]), np.array([0.01058760, 0.59516701, 0.46200101]))],
    12: [(0, np.array([299.237414, 54.506468, 14.751578]), np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([15.121824, 3.513987, 1.142857]), np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([15.121824, 3.513987, 1.142857]), np.array([0.15591627, 0.60768372, 0.39195739])),
         (0, np.array([1.395448, 0.389327, 0.152380]), np.array([-0.21962037, 0.22559543, 0.90039843])),
         (1, np.array([1.395448, 0.389327, 0.152380]), np.array([0.01058760, 0.59516701, 0.46200101]))],
    13: [(0, np.array([351.421477, 64.011861, 17.324108]), np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([18.899396, 4.391813, 1.428354]), np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([18.899396, 4.391813, 1.428354]), np.array([0.15591627, 0.60768372, 0.39195739])),
         (0, np.array([1.395448, 0.389327, 0.152380]), np.array([-0.21962037, 0.22559543, 0.90039843])),
         (1, np.array([1.395448, 0.389327, 0.152380]), np.array([0.01058760, 0.59516701, 0.46200101]))],
    14: [(0, np.array([407.797551, 74.280833, 20.103292]), np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([23.193656, 5.389707, 1.752900]), np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([23.193656, 5.389707, 1.752900]), np.array([0.15591627, 0.60768372, 0.39195739])),
         (0, np.array([1.478741, 0.412565, 0.161475]), np.array([-0.21962037, 0.22559543, 0.90039843])),
         (1, np.array([1.478741, 0.412565, 0.161475]), np.array([0.01058760, 0.59516701, 0.46200101]))],
    15: [(0, np.array([468.365638, 85.313386, 23.089132]), np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([28.032640, 6.514183, 2.118614]), np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([28.032640, 6.514183, 2.118614]), np.array([0.15591627, 0.60768372, 0.39195739])),
         (0, np.array([1.743103, 0.486321, 0.190343]), np.array([-0.21962037, 0.22559543, 0.90039843])),
         (1, np.array([1.743103, 0.486321, 0.190343]), np.array([0.01058760, 0.59516701, 0.46200101]))],
    16: [(0, np.array([533.125736, 97.109518, 26.281625]), np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([33.329752, 7.745118, 2.518953]), np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([33.329752, 7.745118, 2.518953]), np.array([0.15591627, 0.60768372, 0.39195739])),
         (0, np.array([2.029194, 0.566140, 0.221583]), np.array([-0.21962037, 0.22559543, 0.90039843])),
         (1, np.array([2.029194, 0.566140, 0.221583]), np.array([0.01058760, 0.59516701, 0.46200101]))],
    17: [(0, np.array([601.345614, 109.535854, 29.644677]), np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([38.960419, 9.053563, 2.944500]), np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([38.960419, 9.053563, 2.944500]), np.array([0.15591627, 0.60768372, 0.39195739])),
         (0, np.array([2.129386, 0.594093, 0.232524]), np.array([-0.21962037, 0.22559543, 0.90039843])),
         (1, np.array([2.129386, 0.594093, 0.232524]), np.array([0.01058760, 0.59516701, 0.46200101]))],
    18: [(0, np.array([674.446518, 122.851275, 33.248349]), np.array([0.15432897, 0.53532814, 0.44463454])),
         (0, np.array([45.164244, 10.495199, 3.413364]), np.array([-0.09996723, 0.39951283, 0.70011547])),
         (1, np.array([45.164244, 10.495199, 3.413364]), np.array([0.15591627, 0.60768372, 0.39195739])),
         (0, np.array([2.621367, 0.731355, 0.286247]), np.array([-0.21962037, 0.22559543, 0.90039843])),
         (1, np.array([2.621367, 0.731355, 0.286247]), np.array([0.01058760, 0.59516701, 0.46200101]))],
})

# STO-3G fourth-period elements (K-Kr) — shared contraction coefficients
_d1s = np.array([0.15432897, 0.53532814, 0.44463454])   # 1s
_d2s = np.array([-0.09996723, 0.39951283, 0.70011547])   # 2sp s-part
_d2p = np.array([0.15591627, 0.60768372, 0.39195739])    # 2sp p-part
_d3s = np.array([-0.21962037, 0.22559543, 0.90039843])   # 3sp s-part (row 3)
_d3p = np.array([0.01058760, 0.59516701, 0.46200101])    # 3sp p-part (row 3)
_d3s2 = np.array([-0.22776350, 0.21754360, 0.91667696])  # 3sp s-part (Sc-Kr variant)
_d3p2 = np.array([0.00495151, 0.57776647, 0.48464604])   # 3sp p-part (Sc-Kr variant)
_d4s = np.array([-0.30884412, 0.01960641, 1.13103444])   # 4sp s-part
_d4p = np.array([-0.12154686, 0.57152276, 0.54989495])   # 4sp p-part
_dd = np.array([0.21976795, 0.65554736, 0.28657326])     # 3d

def _row4(e1s, e2sp, e3sp, e4sp, ed=None, d3s_v=_d3s, d3p_v=_d3p):
    e1s, e2sp, e3sp, e4sp = np.array(e1s), np.array(e2sp), np.array(e3sp), np.array(e4sp)
    shells = [(0, e1s, _d1s), (0, e2sp, _d2s), (1, e2sp, _d2p),
              (0, e3sp, d3s_v), (1, e3sp, d3p_v), (0, e4sp, _d4s), (1, e4sp, _d4p)]
    if ed is not None:
        shells.insert(5, (2, np.array(ed), _dd))
    return shells

# fmt: (Z, e1s, e2sp, e3sp, e4sp, [ed, d3s_variant, d3p_variant])
_ROW4_DATA = [
    (19, [771.510368,140.531577,38.033329], [52.402040,12.177107,3.960373], [3.651584,1.018783,0.398745], [0.503982,0.186001,0.082140]),
    (20, [854.032495,155.563085,42.101442], [59.560299,13.840533,4.501371], [4.374706,1.220532,0.477708], [0.455849,0.168237,0.074295]),
    (21, [941.662425,171.524986,46.421355], [67.176688,15.610418,5.076992], [4.698159,1.433088,0.552930], [0.630933,0.232854,0.102831], [0.551700,0.168286,0.064930]),
    (22, [1033.57125,188.26629,50.95221], [75.25120,17.48676,5.68724], [5.39554,1.64581,0.63500], [0.71226,0.26287,0.11609], [1.64598,0.50208,0.19372]),
    (23, [1130.76252,205.96980,55.74347], [83.78385,19.46956,6.33211], [6.14115,1.87325,0.72276], [0.71226,0.26287,0.11609], [2.96482,0.90436,0.34893]),
    (24, [1232.32045,224.46871,60.74999], [92.77462,21.55883,7.01160], [6.89949,2.10456,0.81201], [0.75478,0.27856,0.12302], [4.24148,1.29379,0.49918]),
    (25, [1337.15327,243.56414,65.91796], [102.02200,23.70772,7.71049], [7.70196,2.34934,0.90645], [0.67098,0.24763,0.10936], [5.42695,1.65539,0.63870]),
    (26, [1447.40041,263.64579,71.35284], [111.91949,26.00768,8.45851], [8.54857,2.60759,1.00609], [0.59212,0.21853,0.09650], [6.41180,1.95580,0.75461]),
    (27, [1560.83467,284.30798,76.94484], [122.27510,28.41410,9.24115], [9.43931,2.87929,1.11092], [0.59212,0.21853,0.09650], [7.66453,2.33793,0.90204]),
    (28, [1679.77103,305.97239,82.80807], [132.85889,30.87355,10.04104], [10.33074,3.15121,1.21583], [0.63093,0.23285,0.10283], [8.62772,2.63173,1.01540]),
    (29, [1801.80673,328.20135,88.82409], [144.12122,33.49067,10.89221], [11.30775,3.44923,1.33082], [0.63093,0.23285,0.10283], [9.64791,2.94292,1.13547]),
    (30, [1929.43230,351.44850,95.11568], [155.84168,36.21425,11.77800], [12.28153,3.74626,1.44542], [0.88971,0.32836,0.14501], [10.94737,3.33930,1.28840]),
    (31, [2061.42453,375.49105,101.62253], [167.76187,38.98425,12.67889], [12.61506,3.84799,1.48468], [0.79852,0.29471,0.13015], [12.61506,3.84799,1.48468]),
    (32, [2196.38423,400.07413,108.27567], [180.38904,41.91853,13.63321], [14.19666,4.33043,1.67082], [0.98583,0.36383,0.16067], [14.19666,4.33043,1.67082]),
    (33, [2337.06567,425.69943,115.21088], [193.19705,44.89484,14.60120], [15.87164,4.84135,1.86795], [1.10768,0.40880,0.18053], [15.87164,4.84135,1.86795]),
    (34, [2480.62681,451.84927,122.28805], [206.15788,47.90666,15.58073], [17.63999,5.38076,2.07606], [1.21464,0.44828,0.19797], [17.63999,5.38076,2.07606]),
    (35, [2629.99747,479.05732,129.65161], [219.83503,51.08493,16.61441], [19.50173,5.94865,2.29517], [1.39604,0.51523,0.22753], [19.50173,5.94865,2.29517]),
    (36, [2782.16006,506.77393,137.15280], [233.95141,54.36528,17.68128], [21.45685,6.54502,2.52527], [1.59005,0.58683,0.25915], [21.45685,6.54502,2.52527]),
]
for _r in _ROW4_DATA:
    Z, e1s, e2sp, e3sp, e4sp = _r[0], _r[1], _r[2], _r[3], _r[4]
    ed = _r[5] if len(_r) > 5 else None
    d3sv = _d3s if Z <= 20 else _d3s2
    d3pv = _d3p if Z <= 20 else _d3p2
    _STO3G[Z] = _row4(e1s, e2sp, e3sp, e4sp, ed, d3sv, d3pv)

_CCPVDZ: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {
    1: [(0, np.array([13.01, 1.962, 0.4446, 0.122]),
            np.array([0.019685, 0.137977, 0.478148, 0.501240])),
        (0, np.array([0.122]), np.array([1.0])),
        (1, np.array([0.727]), np.array([1.0]))],
    2: [(0, np.array([38.36, 5.770, 1.240, 0.2976]),
            np.array([0.023809, 0.155180, 0.469958, 0.513026])),
        (0, np.array([0.2976]), np.array([1.0])),
        (1, np.array([1.275]), np.array([1.0]))],
    6: [(0, np.array([6665., 1000., 228.0, 64.71, 21.06, 7.495, 2.797, 0.5215, 0.1596]),
            np.array([0.000692, 0.005329, 0.027077, 0.101718, 0.274740, 0.448564, 0.285074, 0.015204, -0.003191])),
        (0, np.array([6665., 1000., 228.0, 64.71, 21.06, 7.495, 2.797, 0.5215, 0.1596]),
            np.array([-0.000146, -0.001154, -0.005725, -0.023312, -0.063955, -0.149981, -0.127262, 0.544529, 0.580496])),
        (0, np.array([0.1596]), np.array([1.0])),
        (1, np.array([9.439, 2.002, 0.5456, 0.1517]),
            np.array([0.038109, 0.209480, 0.508557, 0.468842])),
        (1, np.array([0.1517]), np.array([1.0])),
        (2, np.array([0.55]), np.array([1.0]))],
    7: [(0, np.array([9046., 1357., 309.3, 87.73, 28.56, 10.21, 3.838, 0.7466, 0.2248]),
            np.array([0.000700, 0.005389, 0.027406, 0.103207, 0.278723, 0.448540, 0.278238, 0.015440, -0.002864])),
        (0, np.array([9046., 1357., 309.3, 87.73, 28.56, 10.21, 3.838, 0.7466, 0.2248]),
            np.array([-0.000153, -0.001208, -0.005992, -0.024544, -0.067459, -0.158078, -0.121831, 0.549003, 0.578815])),
        (0, np.array([0.2248]), np.array([1.0])),
        (1, np.array([13.55, 2.917, 0.7973, 0.2185]),
            np.array([0.039919, 0.217169, 0.510319, 0.462214])),
        (1, np.array([0.2185]), np.array([1.0])),
        (2, np.array([0.817]), np.array([1.0]))],
    8: [(0, np.array([11720., 1759., 400.8, 113.7, 37.03, 13.27, 5.025, 1.013, 0.3023]),
            np.array([0.000710, 0.005470, 0.027837, 0.104800, 0.283062, 0.448719, 0.270952, 0.015458, -0.002585])),
        (0, np.array([11720., 1759., 400.8, 113.7, 37.03, 13.27, 5.025, 1.013, 0.3023]),
            np.array([-0.000160, -0.001263, -0.006267, -0.025716, -0.070924, -0.165411, -0.116955, 0.557368, 0.572759])),
        (0, np.array([0.3023]), np.array([1.0])),
        (1, np.array([17.70, 3.854, 1.046, 0.2753]),
            np.array([0.043018, 0.228913, 0.508728, 0.460531])),
        (1, np.array([0.2753]), np.array([1.0])),
        (2, np.array([1.185]), np.array([1.0]))],
}

_BASIS_SETS = {'sto-3g': _STO3G, '6-31g': _631G, 'cc-pvdz': _CCPVDZ}
_ATOMIC_NUMBER = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
    'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
    'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25,
    'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30,
    'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36,
}


def _build_basis(symbols: list[str], coords: np.ndarray, basis_name: str = 'sto-3g') -> list[tuple]:
    if basis_name not in _BASIS_SETS:
        raise ValueError(f"Unknown basis {basis_name!r}. Available: {list(_BASIS_SETS)}")
    basis_data = _BASIS_SETS[basis_name]
    basis = []
    for i, sym in enumerate(symbols):
        Z = _ATOMIC_NUMBER.get(sym.upper())
        if Z is None:
            raise ValueError(f"Unsupported element {sym!r}. Supported: H-Kr (Z=1-36)")
        if Z not in basis_data:
            raise ValueError(f"Element {sym!r} not available in {basis_name}. Try 'sto-3g'.")
        center = coords[i]
        for ang_mom, exps, coeffs in basis_data[Z]:
            if ang_mom == 0:
                norms = (2 * exps / _pi) ** 0.75
                basis.append((center, (0, 0, 0), exps, coeffs * norms))
            elif ang_mom == 1:
                norms = (128 * exps**5 / _pi**3) ** 0.25
                nc = coeffs * norms
                basis.append((center, (1, 0, 0), exps, nc))
                basis.append((center, (0, 1, 0), exps, nc))
                basis.append((center, (0, 0, 1), exps, nc))
            elif ang_mom == 2:
                n_aa = (2048 * exps**7 / (9 * _pi**3)) ** 0.25   # xx, yy, zz
                n_ab = (2048 * exps**7 / _pi**3) ** 0.25          # xy, xz, yz
                for lv in [(2,0,0),(0,2,0),(0,0,2)]:
                    basis.append((center, lv, exps, coeffs * n_aa))
                for lv in [(1,1,0),(1,0,1),(0,1,1)]:
                    basis.append((center, lv, exps, coeffs * n_ab))
    return basis


# Boys function -------

def _boys_array(n_max: int, T: float) -> np.ndarray:
    result = np.zeros(n_max + 1)
    if T < 1e-15:
        for n in range(n_max + 1):
            result[n] = 1.0 / (2 * n + 1)
        return result
    # NOTE: For large T, exp(-T) underflows and kills the downward recursion.
    # Use asymptotic F_0 = sqrt(pi/(4T)) + stable upward recursion instead.
    if T >= 30.0:
        result[0] = (_pi / (4.0 * T)) ** 0.5
        exp_T = np.exp(-T) if T < 700 else 0.0
        for n in range(n_max):
            result[n + 1] = ((2 * n + 1) * result[n] - exp_T) / (2 * T)
        return result
    # Small/moderate T: downward recursion from high-N starting value
    N = max(n_max, int(T)) + 50
    term = 1.0 / (2 * N + 1)
    F_sum = term
    for k in range(1, 400):
        term *= T / (2 * N + 2 * k + 1)
        F_sum += term
        if abs(term) < 1e-15 * abs(F_sum):
            break
    F = F_sum * np.exp(-T)
    exp_T = np.exp(-T)
    for k in range(N - 1, -1, -1):
        F = (2 * T * F + exp_T) / (2 * k + 1)
        if k <= n_max:
            result[k] = F
    return result


# McMurchie-Davidson E coefficients (iterative) -------

def _E_coeffs(l1: int, l2: int, PA: float, PB: float, one_over_2p: float) -> list[float]:
    max_t = l1 + l2
    E = [0.0] * (max_t + 2)  # pad for t+1 access
    E[0] = 1.0
    # Build up i: E^{i+1,0} from E^{i,0}
    for i in range(l1):
        E_new = [0.0] * (max_t + 2)
        for t in range(i + 2):
            E_new[t] = PA * E[t] + (t + 1) * E[t + 1]
            if t > 0:
                E_new[t] += one_over_2p * E[t - 1]
        E = E_new
    # Build up j: E^{l1,j+1} from E^{l1,j}
    for j in range(l2):
        E_new = [0.0] * (max_t + 2)
        for t in range(l1 + j + 2):
            E_new[t] = PB * E[t] + (t + 1) * E[t + 1]
            if t > 0:
                E_new[t] += one_over_2p * E[t - 1]
        E = E_new
    return E[:max_t + 1]


# Hermite Coulomb integrals R^0_{tuv} (iterative) -------

def _R_table(t_max: int, u_max: int, v_max: int, p: float, RPC: np.ndarray, boys_vals: np.ndarray) -> np.ndarray:
    N = t_max + u_max + v_max
    # Step 1: R_t[t][n] for u=v=0
    R_t = [[0.0] * (N + 2) for _ in range(t_max + 1)]
    neg2p = -2.0 * p
    pw = 1.0
    for n in range(N + 1):
        R_t[0][n] = pw * boys_vals[n]
        pw *= neg2p
    x, y, z = float(RPC[0]), float(RPC[1]), float(RPC[2])
    for t in range(t_max):
        Rt1, Rt, Rt_1 = R_t[t + 1], R_t[t], R_t[t - 1] if t > 0 else None
        for n in range(N - t):
            Rt1[n] = x * Rt[n + 1]
        if Rt_1:
            for n in range(N - t):
                Rt1[n] += t * Rt_1[n + 1]
    # Step 2: R_tu[t][u][n] for v=0
    R_tu = [[[0.0] * (N + 2) for _ in range(u_max + 1)] for _ in range(t_max + 1)]
    for t in range(t_max + 1):
        R_tu[t][0] = R_t[t]
    for u in range(u_max):
        for t in range(t_max + 1):
            lim = N - t - u
            if lim <= 0:
                continue
            Ru1, Ru, Ru_1 = R_tu[t][u + 1], R_tu[t][u], R_tu[t][u - 1] if u > 0 else None
            for n in range(lim):
                Ru1[n] = y * Ru[n + 1]
            if Ru_1:
                for n in range(lim):
                    Ru1[n] += u * Ru_1[n + 1]
    # Step 3: R[t][u][v] at n=0
    result = np.zeros((t_max + 1, u_max + 1, v_max + 1))
    for t in range(t_max + 1):
        for u in range(u_max + 1):
            Rv_prev = [0.0] * (N + 2)  # v-1
            Rv_cur = list(R_tu[t][u])   # v=0
            result[t, u, 0] = Rv_cur[0]
            for v in range(v_max):
                lim = N - t - u - v
                if lim <= 0:
                    break
                Rv_next = [0.0] * (N + 2)
                for n in range(lim):
                    Rv_next[n] = z * Rv_cur[n + 1]
                if v > 0:
                    for n in range(lim):
                        Rv_next[n] += v * Rv_prev[n + 1]
                result[t, u, v + 1] = Rv_next[0]
                Rv_prev, Rv_cur = Rv_cur, Rv_next
    return result


# Primitive integrals -------

def _overlap_1d(l1: int, l2: int, PA: float, PB: float, gamma: float) -> float:
    S = [[0.0] * (l2 + 2) for _ in range(l1 + 2)]
    S[0][0] = 1.0
    for i in range(l1):
        S[i + 1][0] = PA * S[i][0]
        if i > 0:
            S[i + 1][0] += i / (2 * gamma) * S[i - 1][0]
    for j in range(l2):
        for i in range(l1 + 1):
            S[i][j + 1] = PB * S[i][j]
            if i > 0:
                S[i][j + 1] += i / (2 * gamma) * S[i - 1][j]
            if j > 0:
                S[i][j + 1] += j / (2 * gamma) * S[i][j - 1]
    return S[l1][l2]


def _overlap_prim(la: tuple, alpha: float, A: np.ndarray, lb: tuple, beta: float, B: np.ndarray) -> float:
    gamma = alpha + beta
    rg = 1.0 / gamma
    Px, Py, Pz = (alpha*A[0]+beta*B[0])*rg, (alpha*A[1]+beta*B[1])*rg, (alpha*A[2]+beta*B[2])*rg
    AB2 = (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2
    prefactor = _exp(-alpha * beta * rg * AB2) * (_pi * rg) ** 1.5
    sx = _overlap_1d(la[0], lb[0], Px - A[0], Px - B[0], gamma)
    sy = _overlap_1d(la[1], lb[1], Py - A[1], Py - B[1], gamma)
    sz = _overlap_1d(la[2], lb[2], Pz - A[2], Pz - B[2], gamma)
    return prefactor * sx * sy * sz


def _kinetic_prim(la: tuple, alpha: float, A: np.ndarray, lb: tuple, beta: float, B: np.ndarray) -> float:
    # T = -1/2 <a|nabla^2|b>, dimension-by-dimension via overlap recurrence
    gamma = alpha + beta
    rg = 1.0 / gamma
    Px, Py, Pz = (alpha*A[0]+beta*B[0])*rg, (alpha*A[1]+beta*B[1])*rg, (alpha*A[2]+beta*B[2])*rg
    PA = (Px - A[0], Py - A[1], Pz - A[2])
    PB = (Px - B[0], Py - B[1], Pz - B[2])
    AB2 = (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2
    prefactor = _exp(-alpha * beta * rg * AB2) * (_pi * rg) ** 1.5
    T = 0.0
    for d in range(3):
        s_other = 1.0
        for k in range(3):
            if k != d:
                s_other *= _overlap_1d(la[k], lb[k], PA[k], PB[k], gamma)
        s_ij = _overlap_1d(la[d], lb[d], PA[d], PB[d], gamma)
        s_ij_p2 = _overlap_1d(la[d], lb[d] + 2, PA[d], PB[d], gamma)
        term = beta * (2 * lb[d] + 1) * s_ij - 2 * beta ** 2 * s_ij_p2
        if lb[d] >= 2:
            s_ij_m2 = _overlap_1d(la[d], lb[d] - 2, PA[d], PB[d], gamma)
            term -= 0.5 * lb[d] * (lb[d] - 1) * s_ij_m2
        T += term * s_other
    return prefactor * T


def _nuclear_prim(la: tuple, alpha: float, A: np.ndarray, lb: tuple, beta: float, B: np.ndarray, C: np.ndarray) -> float:
    gamma = alpha + beta
    rg = 1.0 / gamma
    Px = (alpha * A[0] + beta * B[0]) * rg
    Py = (alpha * A[1] + beta * B[1]) * rg
    Pz = (alpha * A[2] + beta * B[2]) * rg
    PCx, PCy, PCz = Px - C[0], Py - C[1], Pz - C[2]
    AB2 = (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2
    T = gamma * (PCx*PCx + PCy*PCy + PCz*PCz)
    L = sum(la) + sum(lb)
    boys_vals = _boys_array(L, T)
    prefactor = _exp(-alpha * beta * rg * AB2) * 2 * _pi * rg
    h2g = 0.5 * rg
    Ex = _E_coeffs(la[0], lb[0], Px - A[0], Px - B[0], h2g)
    Ey = _E_coeffs(la[1], lb[1], Py - A[1], Py - B[1], h2g)
    Ez = _E_coeffs(la[2], lb[2], Pz - A[2], Pz - B[2], h2g)
    R = _R_table(la[0] + lb[0], la[1] + lb[1], la[2] + lb[2], gamma,
                 np.array([PCx, PCy, PCz]), boys_vals)
    val = 0.0
    for t, ex in enumerate(Ex):
        for u, ey in enumerate(Ey):
            for v, ez in enumerate(Ez):
                val += ex * ey * ez * R[t, u, v]
    return prefactor * val


def _eri_prim(la: tuple, alpha: float, A: np.ndarray, lb: tuple, beta: float, B: np.ndarray,
              lc: tuple, gamma_c: float, C: np.ndarray, ld: tuple, delta_d: float, D: np.ndarray) -> float:
    p = alpha + beta
    q = gamma_c + delta_d
    rp, rq = 1.0 / p, 1.0 / q
    # Gaussian product centers (scalar to avoid numpy array overhead)
    Px = (alpha * A[0] + beta * B[0]) * rp
    Py = (alpha * A[1] + beta * B[1]) * rp
    Pz = (alpha * A[2] + beta * B[2]) * rp
    Qx = (gamma_c * C[0] + delta_d * D[0]) * rq
    Qy = (gamma_c * C[1] + delta_d * D[1]) * rq
    Qz = (gamma_c * C[2] + delta_d * D[2]) * rq
    AB2 = (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2
    CD2 = (C[0]-D[0])**2 + (C[1]-D[1])**2 + (C[2]-D[2])**2
    PQx, PQy, PQz = Px - Qx, Py - Qy, Pz - Qz
    delta = p * q / (p + q)
    T = delta * (PQx*PQx + PQy*PQy + PQz*PQz)
    L = sum(la) + sum(lb) + sum(lc) + sum(ld)
    boys_vals = _boys_array(L, T)
    prefactor = _exp(-alpha * beta * rp * AB2 - gamma_c * delta_d * rq * CD2)
    prefactor *= 2 * _pi ** 2.5 / (p * q * (p + q) ** 0.5)
    h2p, h2q = 0.5 * rp, 0.5 * rq
    Ex1 = _E_coeffs(la[0], lb[0], Px - A[0], Px - B[0], h2p)
    Ey1 = _E_coeffs(la[1], lb[1], Py - A[1], Py - B[1], h2p)
    Ez1 = _E_coeffs(la[2], lb[2], Pz - A[2], Pz - B[2], h2p)
    Ex2 = _E_coeffs(lc[0], ld[0], Qx - C[0], Qx - D[0], h2q)
    Ey2 = _E_coeffs(lc[1], ld[1], Qy - C[1], Qy - D[1], h2q)
    Ez2 = _E_coeffs(lc[2], ld[2], Qz - C[2], Qz - D[2], h2q)
    R = _R_table(la[0]+lb[0]+lc[0]+ld[0], la[1]+lb[1]+lc[1]+ld[1],
                 la[2]+lb[2]+lc[2]+ld[2], delta, np.array([PQx, PQy, PQz]), boys_vals)
    val = 0.0
    for t1, ex1 in enumerate(Ex1):
        for t2, ex2 in enumerate(Ex2):
            for u1, ey1 in enumerate(Ey1):
                for u2, ey2 in enumerate(Ey2):
                    for v1, ez1 in enumerate(Ez1):
                        for v2, ez2 in enumerate(Ez2):
                            val += ex1 * ex2 * ey1 * ey2 * ez1 * ez2 * \
                                   (-1) ** (t2 + u2 + v2) * R[t1 + t2, u1 + u2, v1 + v2]
    return prefactor * val


# Contraction over CGTOs -------

def _compute_ao_integrals(basis: list, symbols: list[str], coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nbf = len(basis)
    S = np.zeros((nbf, nbf))
    T = np.zeros((nbf, nbf))
    V = np.zeros((nbf, nbf))

    for i in range(nbf):
        Ai, lai, exps_i, ci = basis[i]
        for j in range(i + 1):
            Aj, laj, exps_j, cj = basis[j]
            s_val = t_val = v_val = 0.0
            for pi_ in range(len(exps_i)):
                for pj in range(len(exps_j)):
                    c = ci[pi_] * cj[pj]
                    s_val += c * _overlap_prim(lai, exps_i[pi_], Ai, laj, exps_j[pj], Aj)
                    t_val += c * _kinetic_prim(lai, exps_i[pi_], Ai, laj, exps_j[pj], Aj)
                    for k, sym in enumerate(symbols):
                        Z = _ATOMIC_NUMBER[sym.upper()]
                        v_val += c * (-Z) * _nuclear_prim(lai, exps_i[pi_], Ai, laj, exps_j[pj], Aj, coords[k])
            S[i, j] = S[j, i] = s_val
            T[i, j] = T[j, i] = t_val
            V[i, j] = V[j, i] = v_val

    # ERI with inlined contraction and Schwarz screening
    eri = np.zeros((nbf, nbf, nbf, nbf))
    pre_2pi25 = 2.0 * _pi ** 2.5
    # Schwarz bounds (skip for small bases where nothing gets screened)
    Q = None
    if nbf > 10:
        Q = np.zeros((nbf, nbf))
        for i in range(nbf):
            Ai, lai, exps_i, ci = basis[i]
            for j in range(i + 1):
                Aj, laj, exps_j, cj = basis[j]
                diag = 0.0
                for pi_ in range(len(exps_i)):
                    for pj in range(len(exps_j)):
                        for pk in range(len(exps_i)):
                            for pl in range(len(exps_j)):
                                diag += ci[pi_]*cj[pj]*ci[pk]*cj[pl] * _eri_prim(
                                    lai, exps_i[pi_], Ai, laj, exps_j[pj], Aj,
                                    lai, exps_i[pk], Ai, laj, exps_j[pl], Aj)
                Q[i, j] = Q[j, i] = abs(diag) ** 0.5
    for i in range(nbf):
        Ai, lai, exps_i, ci = basis[i]
        Aix, Aiy, Aiz = float(Ai[0]), float(Ai[1]), float(Ai[2])
        for j in range(i + 1):
            Aj, laj, exps_j, cj = basis[j]
            Ajx, Ajy, Ajz = float(Aj[0]), float(Aj[1]), float(Aj[2])
            AB2 = (Aix-Ajx)**2 + (Aiy-Ajy)**2 + (Aiz-Ajz)**2
            ij = i * (i + 1) // 2 + j
            for k in range(nbf):
                Ak, lak, exps_k, ck = basis[k]
                Akx, Aky, Akz = float(Ak[0]), float(Ak[1]), float(Ak[2])
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if kl > ij:
                        continue
                    if Q is not None and Q[i, j] * Q[k, l] < 1e-12:
                        continue
                    Al, lal, exps_l, cl = basis[l]
                    Alx, Aly, Alz = float(Al[0]), float(Al[1]), float(Al[2])
                    CD2 = (Akx-Alx)**2 + (Aky-Aly)**2 + (Akz-Alz)**2
                    L = lai[0]+lai[1]+lai[2]+laj[0]+laj[1]+laj[2]+lak[0]+lak[1]+lak[2]+lal[0]+lal[1]+lal[2]
                    t_max = lai[0]+laj[0]+lak[0]+lal[0]
                    u_max = lai[1]+laj[1]+lak[1]+lal[1]
                    v_max = lai[2]+laj[2]+lak[2]+lal[2]
                    val = 0.0
                    for pi_ in range(len(exps_i)):
                        ai = float(exps_i[pi_]); ci_v = float(ci[pi_])
                        for pj in range(len(exps_j)):
                            bj = float(exps_j[pj]); cj_v = float(cj[pj])
                            p = ai + bj; rp = 1.0 / p
                            Px = (ai*Aix + bj*Ajx)*rp; Py = (ai*Aiy + bj*Ajy)*rp; Pz = (ai*Aiz + bj*Ajz)*rp
                            exp_ab = _exp(-ai * bj * rp * AB2)
                            h2p = 0.5 * rp
                            Ex1 = _E_coeffs(lai[0], laj[0], Px-Aix, Px-Ajx, h2p)
                            Ey1 = _E_coeffs(lai[1], laj[1], Py-Aiy, Py-Ajy, h2p)
                            Ez1 = _E_coeffs(lai[2], laj[2], Pz-Aiz, Pz-Ajz, h2p)
                            cij = ci_v * cj_v * exp_ab
                            for pk in range(len(exps_k)):
                                gk = float(exps_k[pk]); ck_v = float(ck[pk])
                                for pl in range(len(exps_l)):
                                    dl = float(exps_l[pl]); cl_v = float(cl[pl])
                                    q = gk + dl; rq = 1.0 / q
                                    Qx = (gk*Akx + dl*Alx)*rq; Qy = (gk*Aky + dl*Aly)*rq; Qz = (gk*Akz + dl*Alz)*rq
                                    PQx, PQy, PQz = Px-Qx, Py-Qy, Pz-Qz
                                    delta = p * q / (p + q)
                                    T_val = delta * (PQx*PQx + PQy*PQy + PQz*PQz)
                                    boys_vals = _boys_array(L, T_val)
                                    pf = cij * ck_v * cl_v * _exp(-gk * dl * rq * CD2) * pre_2pi25 / (p * q * (p+q)**0.5)
                                    h2q = 0.5 * rq
                                    Ex2 = _E_coeffs(lak[0], lal[0], Qx-Akx, Qx-Alx, h2q)
                                    Ey2 = _E_coeffs(lak[1], lal[1], Qy-Aky, Qy-Aly, h2q)
                                    Ez2 = _E_coeffs(lak[2], lal[2], Qz-Akz, Qz-Alz, h2q)
                                    R = _R_table(t_max, u_max, v_max, delta, np.array([PQx, PQy, PQz]), boys_vals)
                                    pv = 0.0
                                    for t1, ex1 in enumerate(Ex1):
                                        for t2, ex2 in enumerate(Ex2):
                                            for u1, ey1 in enumerate(Ey1):
                                                for u2, ey2 in enumerate(Ey2):
                                                    for v1, ez1 in enumerate(Ez1):
                                                        for v2, ez2 in enumerate(Ez2):
                                                            pv += ex1*ex2*ey1*ey2*ez1*ez2 * (-1)**(t2+u2+v2) * R[t1+t2,u1+u2,v1+v2]
                                    val += pf * pv
                    for a, b, c, d in [(i,j,k,l),(j,i,k,l),(i,j,l,k),(j,i,l,k),
                                        (k,l,i,j),(l,k,i,j),(k,l,j,i),(l,k,j,i)]:
                        eri[a, b, c, d] = val
    return S, T, V, eri


def _nuclear_repulsion(symbols: list[str], coords: np.ndarray) -> float:
    e = 0.0
    for i in range(len(symbols)):
        Zi = _ATOMIC_NUMBER[symbols[i].upper()]
        for j in range(i + 1, len(symbols)):
            Zj = _ATOMIC_NUMBER[symbols[j].upper()]
            e += Zi * Zj / np.linalg.norm(coords[i] - coords[j])
    return e


# RHF-SCF with DIIS -------

def _rhf_core(S, T, V, eri, n_electrons, max_iter, tol, use_diis):
    """Single RHF SCF run. Returns (electronic_energy, MO_coefficients, Fock_matrix)."""
    n_occ = n_electrons // 2
    H_core = T + V
    eigvals, eigvecs = np.linalg.eigh(S)
    X = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T

    F_prime = X.T @ H_core @ X
    _, C_prime = np.linalg.eigh(F_prime)
    C = X @ C_prime
    D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    diis_focks, diis_errors = [], []
    E_old, E_elec = 0.0, 0.0
    for iteration in range(max_iter):
        J = np.einsum('kl,ijkl->ij', D, eri)
        K = np.einsum('kl,ikjl->ij', D, eri)
        F = H_core + J - 0.5 * K

        if use_diis:
            err = F @ D @ S - S @ D @ F
            diis_focks.append(F.copy())
            diis_errors.append(err.copy())
            if len(diis_focks) > 8: diis_focks.pop(0); diis_errors.pop(0)
            if len(diis_focks) >= 2:
                n = len(diis_focks)
                B = np.zeros((n + 1, n + 1))
                B[-1, :] = B[:, -1] = -1.0; B[-1, -1] = 0.0
                for ii in range(n):
                    for jj in range(n):
                        B[ii, jj] = np.sum(diis_errors[ii] * diis_errors[jj])
                rhs = np.zeros(n + 1); rhs[-1] = -1.0
                try:
                    coeffs = np.linalg.solve(B, rhs)
                    F = sum(c * f for c, f in zip(coeffs[:n], diis_focks))
                except np.linalg.LinAlgError:
                    pass

        F_prime = X.T @ F @ X
        _, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime
        D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

        E_elec = 0.5 * np.sum(D_new * (H_core + F))
        if abs(E_elec - E_old) < tol and iteration > 0:
            return E_elec, C, F, True
        E_old = E_elec
        D = D_new

    return E_elec, C, F, False


def _rhf(S: np.ndarray, T: np.ndarray, V: np.ndarray, eri: np.ndarray,
         n_electrons: int, max_iter: int = 200, tol: float = 1e-10) -> tuple[float, np.ndarray, np.ndarray]:
    """Restricted Hartree-Fock SCF. Falls back to no-DIIS only if DIIS didn't converge."""
    E, C, F, converged = _rhf_core(S, T, V, eri, n_electrons, max_iter, tol, use_diis=True)
    if not converged:
        E2, C2, F2, _ = _rhf_core(S, T, V, eri, n_electrons, max_iter, tol, use_diis=False)
        if E2 < E:
            return E2, C2, F2
    return E, C, F


def _uhf(S: np.ndarray, T: np.ndarray, V: np.ndarray, eri: np.ndarray,
         n_electrons: int, max_iter: int = 200, tol: float = 1e-10) -> tuple[float, np.ndarray, np.ndarray]:
    """Unrestricted Hartree-Fock SCF. Returns (electronic_energy, MO_coefficients, Fock_matrix).

    MO coefficients are for the alpha orbitals (used for active space extraction).
    """
    nbf = S.shape[0]
    na = (n_electrons + 1) // 2  # alpha electrons
    nb = n_electrons // 2         # beta electrons
    H_core = T + V
    eigvals, eigvecs = np.linalg.eigh(S)
    X = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T

    F_prime = X.T @ H_core @ X
    _, C_prime = np.linalg.eigh(F_prime)
    C = X @ C_prime
    Da = C[:, :na] @ C[:, :na].T
    Db = C[:, :nb] @ C[:, :nb].T if nb > 0 else np.zeros_like(Da)

    diis_focks_a, diis_focks_b, diis_errors = [], [], []
    E_old = 0.0
    for iteration in range(max_iter):
        D = Da + Db
        Ja = np.einsum('kl,ijkl->ij', Da, eri)
        Jb = np.einsum('kl,ijkl->ij', Db, eri)
        Ka = np.einsum('kl,ikjl->ij', Da, eri)
        Kb = np.einsum('kl,ikjl->ij', Db, eri)
        Fa = H_core + Ja + Jb - Ka
        Fb = H_core + Ja + Jb - Kb

        err_a = Fa @ Da @ S - S @ Da @ Fa
        err_b = Fb @ Db @ S - S @ Db @ Fb
        diis_focks_a.append(Fa.copy()); diis_focks_b.append(Fb.copy())
        diis_errors.append(np.concatenate([err_a.ravel(), err_b.ravel()]))
        if len(diis_focks_a) > 8:
            diis_focks_a.pop(0); diis_focks_b.pop(0); diis_errors.pop(0)

        if len(diis_focks_a) >= 2:
            n = len(diis_focks_a)
            B = np.zeros((n + 1, n + 1))
            B[-1, :] = B[:, -1] = -1.0
            B[-1, -1] = 0.0
            for ii in range(n):
                for jj in range(n):
                    B[ii, jj] = diis_errors[ii] @ diis_errors[jj]
            rhs = np.zeros(n + 1); rhs[-1] = -1.0
            try:
                coeffs = np.linalg.solve(B, rhs)
                Fa = sum(c * f for c, f in zip(coeffs[:n], diis_focks_a))
                Fb = sum(c * f for c, f in zip(coeffs[:n], diis_focks_b))
            except np.linalg.LinAlgError:
                pass

        _, Ca_prime = np.linalg.eigh(X.T @ Fa @ X)
        Ca = X @ Ca_prime
        Da_new = Ca[:, :na] @ Ca[:, :na].T
        _, Cb_prime = np.linalg.eigh(X.T @ Fb @ X)
        Cb = X @ Cb_prime
        Db_new = Cb[:, :nb] @ Cb[:, :nb].T if nb > 0 else np.zeros_like(Da)

        D_new = Da_new + Db_new
        E_elec = 0.5 * (np.sum(Da_new * (H_core + Fa)) + np.sum(Db_new * (H_core + Fb)))
        if abs(E_elec - E_old) < tol and iteration > 0:
            return E_elec, Ca, Fa
        E_old = E_elec
        Da, Db = Da_new, Db_new

    return E_elec, Ca, Fa


# Active space extraction -------

def _active_space_integrals(C: np.ndarray, H_core: np.ndarray, eri: np.ndarray,
                            n_electrons: int, act_el: int, act_orb: int):
    nf = n_electrons // 2 - act_el // 2  # n_frozen
    a0, a1 = nf, nf + act_orb  # active window

    h1_mo = C.T @ H_core @ C
    eri_mo = np.einsum('pi,qj,pqrs,rk,sl->ijkl', C, C, eri, C, C, optimize=True)

    # Frozen core energy
    e_core = 0.0
    if nf > 0:
        e_core = 2.0 * np.trace(h1_mo[:nf, :nf])
        for i in range(nf):
            for j in range(nf):
                e_core += 2.0 * eri_mo[i, i, j, j] - eri_mo[i, j, j, i]

    # Effective 1-electron integrals in active space
    a = slice(a0, a1)
    h1_act = h1_mo[a, a].copy()
    if nf > 0:
        for k in range(nf):
            h1_act += 2.0 * eri_mo[a0:a1, a0:a1, k, k] - eri_mo[a0:a1, k, k, a0:a1]

    h2_act = eri_mo[a, a, a, a].copy()
    return h1_act, h2_act, e_core


# Public entry point -------

def compute_molecular_integrals(symbols: list[str], geometry: np.ndarray,
                                active_electrons: int | None = None,
                                active_orbitals: int | None = None,
                                basis_name: str = 'sto-3g') -> tuple[np.ndarray, np.ndarray, float, int]:
    """Compute molecular integrals from atoms + coordinates (Bohr).

    Returns (h1_spin, h2_spin, nuclear_repulsion + core_energy, n_active_electrons).
    h1, h2 are in spin-orbital basis, chemist notation.
    """
    basis = _build_basis(symbols, geometry, basis_name)
    S, T, V, eri = _compute_ao_integrals(basis, symbols, geometry)
    n_el = sum(_ATOMIC_NUMBER[s.upper()] for s in symbols)
    nuc = _nuclear_repulsion(symbols, geometry)

    scf = _rhf if n_el % 2 == 0 else _uhf
    E_elec, C, F = scf(S, T, V, eri, n_el)

    nbf = len(basis)
    act_el = active_electrons if active_electrons is not None else n_el
    act_orb = active_orbitals if active_orbitals is not None else nbf

    H_core = T + V
    h1_sp, h2_sp, e_core = _active_space_integrals(C, H_core, eri, n_el, act_el, act_orb)

    # Spatial → spin-orbital (vectorized)
    n_spin = 2 * act_orb
    idx = np.arange(n_spin)
    spin_mask = (idx[:, None] % 2 == idx[None, :] % 2)  # same-spin mask
    orb = idx // 2
    h1 = h1_sp[np.ix_(orb, orb)] * spin_mask
    h2 = h2_sp[np.ix_(orb, orb, orb, orb)] * (spin_mask[:, :, None, None] & spin_mask[None, None, :, :])

    return h1, h2, nuc + e_core, act_el
