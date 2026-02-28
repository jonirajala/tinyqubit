import numpy as np
from tinyqubit.qml import quantum_kernel, kernel_matrix
from tinyqubit.qml.feature_map import angle_feature_map, basis_feature_map, zz_feature_map


def test_kernel_self():
    x = [0.5, 1.2]
    assert np.isclose(quantum_kernel(zz_feature_map, x, x), 1.0)


def test_kernel_symmetric():
    x, y = [0.3, 0.7], [1.1, 2.0]
    assert np.isclose(quantum_kernel(zz_feature_map, x, y), quantum_kernel(zz_feature_map, y, x))


def test_kernel_range():
    x, y = [0.1, 0.9], [2.5, 1.3]
    k = quantum_kernel(zz_feature_map, x, y)
    assert 0.0 <= k <= 1.0 + 1e-10


def test_kernel_orthogonal():
    assert quantum_kernel(basis_feature_map, [0], [1]) < 1e-10


def test_kernel_matrix_shape():
    X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    K = kernel_matrix(angle_feature_map, X)
    assert K.shape == (3, 3)


def test_kernel_matrix_symmetric():
    X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    K = kernel_matrix(zz_feature_map, X)
    np.testing.assert_allclose(K, K.T)


def test_kernel_matrix_diagonal():
    X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    K = kernel_matrix(zz_feature_map, X)
    np.testing.assert_allclose(np.diag(K), 1.0)


def test_kernel_matrix_rectangular():
    X1 = [[0.1, 0.2], [0.3, 0.4]]
    X2 = [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
    K = kernel_matrix(angle_feature_map, X1, X2)
    assert K.shape == (2, 3)


def test_kernel_matrix_deterministic():
    X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    K1 = kernel_matrix(zz_feature_map, X)
    K2 = kernel_matrix(zz_feature_map, X)
    np.testing.assert_array_equal(K1, K2)


def test_kernel_different_feature_maps():
    x, y = [0.5, 1.0], [1.5, 2.0]
    for fm in [angle_feature_map, zz_feature_map]:
        k = quantum_kernel(fm, x, y)
        assert 0.0 <= k <= 1.0 + 1e-10
