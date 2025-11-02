# -*- coding: utf-8 -*-

import numpy as np

from electrotechnics.circuit import solve_circuit, report
from electrotechnics.plotting import phasor_plot

# -----------------------------
# Example circuit definition
# -----------------------------
# Nodes: 0 (ground), 1, 2, 3
# Solve the circuit and make a phasor diagram (plot).

diagram = """
     |-------[ R3 ]-------(3)-------[ R4 ]----------|
     |                     |                        |
     |                     |                        |
     |                     |                        |
     |                   [ R2 ]                     |
     |                     |                        |
     |                     |                        |
     |                     |                        |
    (0)------[ R1 ]-------(1)------[  L_12 ]-------(2)
     |                                              |
     |                                              |
     |                                              |
     |----------------[  C_20 ]---------------------|
     |                                              |
     |                                              |
     |------------------(+ V source -) -------------|

R1 = 100 Ohm
R2 = 200 Ohm
R3 = 50 Ohm
R4 = 150 Ohm
L12 = 10e-3 H
C20 = 10e-6 F

frequency = 50 Hz

Gound node: 0
Voltage source: 220 V between nodes 0 and 2
"""

print(diagram)

f = 50.0  # Hz
w = 2 * np.pi * f

# Components:
R1 = 100.0      # Ohm between 1-0
R2 = 200.0      # Ohm between 1-3
R3 = 50.0       # Ohm between 3-0
R4 = 150.0      # Ohm between 2-3
L12 = 10e-3     # H between 1-2
C20 = 10e-6     # F between 2-0

edges = [
    (1, 0, R1),         # R1 between 1-0
    (1, 2, 1j*w*L12),   # XL between 1-2
    (2, 0, -1j/(w*C20)), # XC between 2-0
    (1, 3, R2),         # R2 between 1-3
    (3, 0, R3),         # R3 between 3-0
    (2, 3, R4),         # R4 between 2-3
]
v_sources = [(0, 2, 220.0)]  # Voltage source: 220 V between nodes 0 and 2
solution = solve_circuit(edges, voltage_sources=v_sources, ground=0)
report_ = report(edges, v_sources, current_sources=[], solution=solution)

for key, value in solution.items():
    print(f"{key}: {value}\n")


# Example: currents I_b0, I_b1, I_vs0 and voltages V_1-0, V_2-0, E_vs0
# 1) Generate the diagram (with tooltips and separate scales)
cur_specs  = [0, 1, ('vs', 0)]
volt_specs = [(1, 0), (2, 0), ('vs', 0)]

phasor_plot(solution=solution,
            currents=cur_specs,
            voltages=volt_specs,
            voltage_sources=v_sources,
            current_sources=[],
            reference=(1, 0),          # ref = V(1-0)
            separate_scales=True,
            hide_axes=True,
            show_grid=False,
            title='Phasor diagram')
