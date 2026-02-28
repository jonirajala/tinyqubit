from .feature_map import angle_feature_map, basis_feature_map, amplitude_feature_map, zz_feature_map, pauli_feature_map
from .ansatz import strongly_entangling_layers, basic_entangler_layers
from .optimize import GradientDescent, Adam, SPSA, QNG
from .kernel import quantum_kernel, kernel_matrix
from .cost import predict, cross_entropy_cost, mse_cost, fidelity_cost
from .hamiltonian import maxcut_hamiltonian
from .trainability import gradient_variance, expressibility
