# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_laplacian_y(edges_y):
    """
    Build the complex Laplacian (nodal admittance) matrix from a list of edges.
    edges_y: list of tuples (n1, n2, y) where y is a complex admittance.
    """
    nodes = [max(edge[:-1]) for edge in edges_y]
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
    nodes = [max(edge[:-1]) for edge in edges_z]
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

def solve_circuit_y(edges_y,
                    voltage_sources=None,
                    current_sources=None,
                    ground=0,
                    return_branch_currents=True,
                    validate_numbering=True):
    """
    Solve a linear circuit (DC or AC phasor) using Modified Nodal Analysis (MNA).
    The number of nodes is inferred from 'edges_y', 'voltage_sources', and 'current_sources'.

    Parameters
    ----------
    edges_y : list[tuple[int,int,complex]]
        Passive branches as (n1, n2, y), where y is the complex admittance (can be 0+0j).
    voltage_sources : list[tuple[int,int,complex]] | None
        Independent voltage sources as (n_plus, n_minus, V). V is phasor voltage (positive from n_plus to n_minus).
    current_sources : list[tuple[int,int,complex]] | None
        Independent current sources as (n_plus, n_minus, I). I is phasor current injected from n_plus to n_minus
        (+I at n_plus, −I at n_minus in KCL).
    ground : int
        Reference node index (V=0). Must exist in the node numbering.
    return_branch_currents : bool
        If True, returns currents through passive branches in the order of edges_y (direction n1->n2).
    validate_numbering : bool
        If True, validates that nodes are numbered consecutively from 0 to N-1.

    Returns
    -------
    dict with:
        'node_voltages' : np.ndarray (N,), node voltages with respect to ground.
        'voltage_source_currents' : np.ndarray (m,), currents through voltage sources (entering circuit at + terminal).
        'branch_currents' : np.ndarray (k,), currents through passive branches (if requested).
        'G_reduced', 'B', 'I_reduced' : internal matrices/vectors for diagnostics.
        'num_nodes' : int, inferred number of nodes.
    """
    voltage_sources = voltage_sources or []
    current_sources = current_sources or []

    # --- Infer set of referenced nodes ---
    nodes = set([ground])
    for n1, n2, _ in edges_y:
        nodes.add(int(n1)); nodes.add(int(n2))
    for np_, nm, _ in voltage_sources:
        nodes.add(int(np_)); nodes.add(int(nm))
    for np_, nm, _ in current_sources:
        nodes.add(int(np_)); nodes.add(int(nm))

    if len(nodes) == 0:
        raise ValueError("No se detectaron nodos en las entradas.")

    max_node = max(nodes)
    min_node = min(nodes)
    num_nodes = max_node + 1

    # --- Optional validation of consecutive numbering 0..N-1 ---
    if validate_numbering:
        if min_node != 0:
            raise ValueError(f"Node numbering must start at 0 (min_node={min_node}).")
        expected = set(range(num_nodes))
        if nodes - expected:
            raise ValueError("Nodes detected outside expected range 0..max.")
        if expected - nodes:
            missing = sorted(list(expected - nodes))
            raise ValueError(f"Missing intermediate nodes for consecutive numbering: {missing}")

    # --- Build list of active nodes (without ground) and index map ---
    keep_nodes = [n for n in range(num_nodes) if n != ground]
    idx_map = {n: k for k, n in enumerate(keep_nodes)}

    # --- Build complete G matrix (Laplacian/admittance) and reduce ---
    G_full = np.zeros((num_nodes, num_nodes), dtype=complex)
    for n1, n2, y in edges_y:
        if n1 == n2:
            G_full[n1, n1] += y
        else:
            G_full[n1, n1] += y
            G_full[n2, n2] += y
            G_full[n1, n2] -= y
            G_full[n2, n1] -= y
    G = G_full[np.ix_(keep_nodes, keep_nodes)]

    # --- Current source injection vector ---
    I = np.zeros((len(keep_nodes),), dtype=complex)
    for np_, nm, Ival in current_sources:
        if np_ != ground:
            I[idx_map[np_]] += Ival
        if nm != ground:
            I[idx_map[nm]] -= Ival

    # --- B matrix / E vector for voltage sources ---
    m = len(voltage_sources)
    if m > 0:
        B = np.zeros((len(keep_nodes), m), dtype=complex)
        E = np.zeros((m,), dtype=complex)
        for j, (np_, nm, Vval) in enumerate(voltage_sources):
            if np_ != ground:
                B[idx_map[np_], j] = 1.0
            if nm != ground:
                B[idx_map[nm], j] = -1.0
            E[j] = Vval

        # Sistema MNA:
        # [ G  B ] [v]   [ I ]
        # [ B^T 0 ] [i] = [ E ]
        A = np.block([
            [G, B],
            [B.conj().T, np.zeros((m, m), dtype=complex)]
        ])
        rhs = np.concatenate([I, E])
        sol = np.linalg.solve(A, rhs)
        v_red = sol[:len(keep_nodes)]
        i_vs = sol[len(keep_nodes):]
    else:
        # No voltage sources, classic nodal system G v = I
        B = np.zeros((len(keep_nodes), 0), dtype=complex)
        E = np.zeros((0,), dtype=complex)
        v_red = np.linalg.solve(G, I) if G.size else np.array([], dtype=complex)
        i_vs = np.zeros((0,), dtype=complex)

    # --- Reconstruct voltages including ground ---
    V = np.zeros((num_nodes,), dtype=complex)
    for n, k in idx_map.items():
        V[n] = v_red[k]
    V[ground] = 0.0

    result = {
        'node_voltages': V,
        'voltage_source_currents': i_vs,
        'G_reduced': G,
        'B': B,
        'I_reduced': I,
        'num_nodes': num_nodes,
    }

    # --- Currents through passive branches (n1 -> n2) ---
    if return_branch_currents:
        Ibranches = [y * (V[n1] - V[n2]) for (n1, n2, y) in edges_y]
        result['branch_currents'] = np.array(Ibranches, dtype=complex)

    return result

def solve_circuit(edges,
                  voltage_sources=None,
                  current_sources=None,
                  ground=0,
                  return_branch_currents=True,
                  validate_numbering=True):
    """
    Solve a linear circuit (DC or AC phasor) using Modified Nodal Analysis (MNA).
    This is a convenience wrapper around solve_circuit_y() that takes impedance values
    instead of admittances.

    Parameters
    ----------
    edges : list[tuple[int,int,complex]]
        Passive branches as (n1, n2, z), where z is the complex impedance.
    voltage_sources : list[tuple[int,int,complex]] | None
        Independent voltage sources as (n_plus, n_minus, V).
    current_sources : list[tuple[int,int,complex]] | None
        Independent current sources as (n_plus, n_minus, I).
    ground : int
        Reference node index (V=0).
    return_branch_currents : bool
        If True, returns currents through passive branches.
    validate_numbering : bool
        If True, validates consecutive node numbering.

    Returns
    -------
    dict
        Same as solve_circuit_y().
    """
    edges_y = []
    for n1, n2, z in edges:
        y = 1 / z
        edges_y.append((n1, n2, y))
    return solve_circuit_y(edges_y,
                           voltage_sources=voltage_sources,
                           current_sources=current_sources,
                           ground=ground,
                           return_branch_currents=return_branch_currents,
                           validate_numbering=validate_numbering)

def report_y(edges_y, voltage_sources, current_sources, solution, to_pandas=True):
    """
    Generate a detailed report of circuit solution including node voltages, branch currents,
    and power calculations.

    Power conventions:
      - VOLTAGE source: S_src = E * conj(I_vs)  [power delivered to circuit]
      - CURRENT source: S_src = V * conj(I)  where V = V(n_plus) - V(n_minus)
      - Passive branch: S_branch = Vdrop * conj(Ibranch)  [power ABSORBED by branch]

    Parameters
    ----------
    edges_y : list[tuple[int,int,complex]]
        Passive branches with admittances.
    voltage_sources : list[tuple[int,int,complex]] | None
        Voltage sources specification.
    current_sources : list[tuple[int,int,complex]] | None
        Current sources specification.
    solution : dict
        Solution dictionary from solve_circuit_y().
    to_pandas : bool, optional
        If True, returns DataFrames. If False, returns lists of dicts.

    Returns
    -------
    dict
        Contains DataFrames or lists with detailed circuit information:
        - nodes: Node voltages
        - passive_branches: Branch currents and powers
        - voltage_sources: Source currents and powers
        - current_sources: Source voltages and powers
        - totals: Power summaries
        - S_totals: Complex power totals
    """
    V = solution['node_voltages']
    num_nodes = solution['num_nodes']
    Ibranches = solution.get('branch_currents', None)
    i_vs = solution.get('voltage_source_currents', np.zeros((0,), dtype=complex))

    # Nodos
    nodes_rows = []
    for n in range(num_nodes):
        v = V[n]
        nodes_rows.append({'node': n, 'V': v,
                           'V_mag': np.abs(v), 'V_ang_deg': np.angle(v, deg=True)})

    # Ramas pasivas
    branches_rows, S_branches_sum = [], 0+0j
    for k, (n1, n2, y) in enumerate(edges_y):
        vdrop = V[n1] - V[n2]
        ibranch = y * vdrop if Ibranches is None else Ibranches[k]
        S = vdrop * np.conj(ibranch)  # absorbida por la rama
        P, Q = S.real, S.imag
        pf = (P/abs(S)) if abs(S) > 1e-12 else float('nan')
        branches_rows.append({
            'branch': k, 'n1': n1, 'n2': n2,
            'y': y, 'Z_equiv': (1/y if y != 0 else np.inf),
            'I': ibranch, 'I_mag': np.abs(ibranch), 'I_ang_deg': np.angle(ibranch, deg=True),
            'Vdrop': vdrop, 'V_mag': np.abs(vdrop), 'V_ang_deg': np.angle(vdrop, deg=True),
            'S': S, 'P_W': P, 'Q_var': Q, 'PF': pf
        })
        S_branches_sum += S

    # Fuentes de tensión
    vsrc_rows, S_vsrc_sum = [], 0+0j
    for j, (np_, nm, E) in enumerate(voltage_sources or []):
        Ivs = i_vs[j]
        S = E * np.conj(Ivs)   # entregada por la fuente al circuito
        S_vsrc_sum += S
        Vactual = V[np_] - V[nm]
        vsrc_rows.append({
            'id': j, 'n_plus': np_, 'n_minus': nm,
            'E': E, 'I_into_circuit': Ivs,
            'S_delivered': S, 'P_W': S.real, 'Q_var': S.imag,
            'V_nodes_check': Vactual
        })

    # Fuentes de corriente
    isrc_rows, S_isrc_sum = [], 0+0j
    for j, (np_, nm, Ival) in enumerate(current_sources or []):
        Vsrc = V[np_] - V[nm]
        S = Vsrc * np.conj(Ival)  # entregada por la fuente al circuito
        S_isrc_sum += S
        isrc_rows.append({
            'id': j, 'n_plus': np_, 'n_minus': nm,
            'I_injected': Ival, 'V_across': Vsrc,
            'S_delivered': S, 'P_W': S.real, 'Q_var': S.imag
        })

    # Totales
    S_sources = S_vsrc_sum + S_isrc_sum
    S_absorbed = S_branches_sum
    S_residual = S_sources - S_absorbed

    totals = [{
        'S_sources': S_sources, 'P_sources_W': S_sources.real, 'Q_sources_var': S_sources.imag,
        'S_absorbed': S_absorbed, 'P_abs_W': S_absorbed.real, 'Q_abs_var': S_absorbed.imag,
        'S_residual': S_residual, 'P_res_W': S_residual.real, 'Q_res_var': S_residual.imag,
    }]

    out = {'S_totals': {'sources': S_sources, 'absorbed': S_absorbed, 'residual': S_residual}}
    if to_pandas:
        out.update({
            'nodes': pd.DataFrame(nodes_rows),
            'passive_branches': pd.DataFrame(branches_rows),
            'voltage_sources': pd.DataFrame(vsrc_rows),
            'current_sources': pd.DataFrame(isrc_rows),
            'totals': pd.DataFrame(totals),
        })
    else:
        out.update({
            'nodes': nodes_rows, 'passive_branches': branches_rows,
            'voltage_sources': vsrc_rows, 'current_sources': isrc_rows,
            'totals': totals,
        })
    return out

def report(edges, voltage_sources, current_sources, solution, to_pandas=True):
    """
    Generate a detailed circuit solution report, taking impedance values instead of admittances.
    This is a convenience wrapper around report_y().

    Parameters
    ----------
    edges : list[tuple[int,int,complex]]
        Passive branches with impedances.
    voltage_sources : list[tuple[int,int,complex]] | None
        Voltage sources specification.
    current_sources : list[tuple[int,int,complex]] | None
        Current sources specification.
    solution : dict
        Solution dictionary from solve_circuit().
    to_pandas : bool, optional
        If True, returns DataFrames. If False, returns lists of dicts.

    Returns
    -------
    dict
        Same as report_y().
    """
    edges_y = []
    for n1, n2, z in edges:
        y = 1 / z
        edges_y.append((n1, n2, y))
    return report_y(edges_y, voltage_sources, current_sources, solution, to_pandas=to_pandas)

def export_report_to_excel(report_dict, filepath,
                           sheet_order=('nodes','passive_branches','voltage_sources',
                                        'current_sources','totals')):
    """
    Export a circuit solution report to an Excel file with multiple sheets.

    Parameters
    ----------
    report_dict : dict
        Report dictionary from report() or report_y().
    filepath : str
        Path where the Excel file will be saved.
    sheet_order : tuple[str], optional
        Order of sheets in the Excel file. Default includes all standard report sections.

    Returns
    -------
    str
        Path to the created Excel file.

    Notes
    -----
    Creates an additional 'complex_summaries' sheet with complex power totals.
    Empty DataFrames are exported as empty sheets for consistency.
    """
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for sheet in sheet_order:
            df = report_dict.get(sheet, pd.DataFrame())
            if isinstance(df, list): df = pd.DataFrame(df)
            if df is None or len(df) == 0: df = pd.DataFrame()
            df.to_excel(writer, index=False, sheet_name=sheet)
        # hoja con complejos totales
        S = report_dict['S_totals']
        dfS = pd.DataFrame({'metric': ['S_sources','S_absorbed','S_residual'],
                            'value': [S['sources'], S['absorbed'], S['residual']]})
        dfS.to_excel(writer, index=False, sheet_name='complex_summaries')
    return filepath
