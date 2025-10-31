# -*- coding: utf-8 -*-

import numpy as np


def build_laplacian_y(edges_y):
    """
    Build the complex Laplacian (nodal admittance) matrix from a list of edges.
    edges_y: list of tuples (n1, n2, y) where y is a complex admittance.
    """
    nodes = [edge[0] for edge in edges_y]
    num_nodes = max(nodes) + 1
    L = np.zeros((num_nodes, num_nodes), dtype=complex)
    for n1, n2, y in edges_y:
        if n1 == n2:
            L[n1, n1] += y
        else:
            L[n1, n1] += y
            L[n2, n2] += y
            L[n1, n2] -= y
            L[n2, n1] -= y
    return L

def build_laplacian(edges_z):
    """
    Build the complex Laplacian (nodal admittance) matrix from a list of edges.
    edges_z: list of tuples (n1, n2, z) where z is a complex impedance.
    """
    nodes = [edge[0] for edge in edges_z]
    num_nodes = max(nodes) + 1
    L = np.zeros((num_nodes, num_nodes), dtype=complex)
    for n1, n2, z in edges_z:
        y = 1 / z
        if n1 == n2:
            L[n1, n1] += y
        else:
            L[n1, n1] += y
            L[n2, n2] += y
            L[n1, n2] -= y
            L[n2, n1] -= y
    return L

def z_th_pseudoinverse(L, a, b):
    """
    Equivalent impedance between nodes a and b using the Moore–Penrose pseudoinverse.
    Z_ab = (e_a - e_b)^H L^+ (e_a - e_b)
    """
    e = np.zeros((L.shape[0], 1), dtype=complex)
    e[a, 0] = 1.0
    e[b, 0] = -1.0
    # Numerically stable pseudoinverse
    L_plus = np.linalg.pinv(L)
    Z = (e.conj().T @ L_plus @ e)[0, 0]
    return Z

def z_th_test_current(L, a, b):
    """
    Equivalent impedance obtained by applying 1 A from a to b (with b as reference).
    Solves Y_red * v_red = i_red with v(b) = 0. Z_ab = V(a) - V(b) = V(a).
    """
    n = L.shape[0]
    keep = [i for i in range(n) if i != b]
    Y_red = L[np.ix_(keep, keep)]
    i_red = np.zeros((n-1,), dtype=complex)
    # Inject +1 A at 'a' if a != b (its index changes in the reduced system)
    idx_map = {node: k for k, node in enumerate(keep)}
    i_red[idx_map[a]] = 1.0
    v_red = np.linalg.solve(Y_red, i_red)
    V_a = v_red[idx_map[a]]
    return V_a

def z_th(L, a, b):
    """
    Equivalent impedance between nodes a and b.
    Uses the pseudoinverse method.
    """
    return z_th_pseudoinverse(L, a, b)

def z_th_circuit(edges_z, a, b):
    """
    Equivalent impedance between nodes a and b of a circuit defined by edges with impedances.
    edges_z: list of tuples (n1, n2, z) where z is a complex impedance.
    """
    L = build_laplacian(edges_z)
    return z_th(L, a, b)
