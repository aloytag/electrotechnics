# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from electrotechnics.circuit import z_th_circuit

# -----------------------------
# Example circuit definition
# -----------------------------
# Nodes: 0 (ground), 1, 2, 3
# Compute Z_th between 1 and 0.

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

R1 = 100 Ohm
R2 = 200 Ohm
R3 = 50 Ohm
R4 = 150 Ohm
L12 = 10e-3 H
C20 = 10e-6 F

frequency = 1000 Hz
"""

print("Circuit diagram:")
print(diagram)

f0 = 1000.0  # Hz
w = 2*np.pi*f0

# Components:
R1 = 100.0      # Ohm between 1-0
R2 = 200.0      # Ohm between 1-3
R3 = 50.0       # Ohm between 3-0
R4 = 150.0      # Ohm between 2-3
L12 = 10e-3     # H between 1-2
C20 = 10e-6     # F between 2-0

edges = [
    (1, 0, R1),   # R1 1-0
    (1, 2, 1j*w*L12),  # XL  1-2
    (2, 0, -1j/(w*C20)),  # XC  2-0
    (1, 3, R2),   # R2 1-3
    (3, 0, R3),   # R3 3-0
    (2, 3, R4),   # R4 2-3
]

zth = z_th_circuit(edges, a=1, b=0)

print(f"Thevenin Impedance Z_th between node 1 and ground (node 0):")
print(f"Complex Z_th = {zth} Ohm")
print(f"Magnitude |Z_th| = {abs(zth)} Ohm")
print(f"Phase ∠Z_th = {np.angle(zth, deg=True)} deg")

# -----------------------------
# Frequency sweep and plotting
# -----------------------------
freqs = np.logspace(1, 5, 400)  # 10 Hz to 100 kHz
Zf = []
for f in freqs:
    w = 2*np.pi*f
    XL_12 = 1j*w*L12
    XC_20 = -1j/(w*C20)
    edges_f = [
        (1, 0, R1),
        (1, 2, XL_12),
        (2, 0, XC_20),
        (1, 3, R2),
        (3, 0, R3),
        (2, 3, R4),
    ]
    Zf.append(z_th_circuit(edges_f, a=1, b=0))

Zf = np.array(Zf)

plt.figure(figsize=(8, 5))
plt.subplot(2,1,1)
plt.semilogx(freqs, np.abs(Zf))
plt.ylabel('|Z_th| [Ohm]')
plt.grid(True, which='both', ls=':')
plt.title('Thevenin impedance seen from node 1 to ground (node 0)')

plt.subplot(2,1,2)
plt.semilogx(freqs, np.angle(Zf, deg=True))
plt.ylabel('Phase [°]')
plt.xlabel('Frequency [Hz]')
plt.grid(True, which='both', ls=':')

plt.tight_layout()
plt.show()
