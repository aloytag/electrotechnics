# Electrotechnics

A Python library for linear circuit analysis using Modified Nodal Analysis (MNA). Supports DC and AC (phasor) analysis.

This repository provides utilities for:
- Building and solving linear circuits with passive elements, voltage and current sources
- Computing Thevenin equivalent impedances between nodes
- Generating detailed reports including voltages, currents, and power analysis
- Interactive phasor diagrams for voltage and current visualization
- Example scripts for frequency response and power flow analysis

## Features

Circuit Analysis:
- Solve linear circuits using Modified Nodal Analysis (MNA)
- Support for:
  - Passive branches (complex impedances/admittances)
  - Independent voltage sources
  - Independent current sources
  - Ground node selection
- Compute node voltages and branch currents
- Power flow analysis (complex power delivered/absorbed)
- Interactive phasor diagrams with:
  - Voltage and current phasors with magnitude/phase tooltips
  - Independent voltage/current scaling
  - Configurable reference angles
  - Toggle controls for visibility

Thevenin Analysis:
- Build nodal Laplacian (admittance) matrices from edge lists (impedance or admittance input).
- Compute equivalent Thevenin impedance between two nodes using the Moore–Penrose pseudoinverse or test-current method.

Reporting:
- Detailed solution reports with:
  - Node voltages (magnitude and phase)
  - Branch currents and powers
  - Source currents and delivered powers
  - Power summaries and conservation checks
- Export reports to Excel workbooks

Examples:
- Frequency response analysis
- Power flow visualization
- Complete circuit solving examples

## Installation

Install the lastest stable version from the PyPI repository.

```
pip install electrotechnics -U
```

On MS Windows you may prefer:

```
python -m pip install electrotechnics -U
```

Once installed, run the example in a Python script or console:

```python
from electrotechnics.examples import example1
```


## Requirements

- Python 3.8+
- numpy
- matplotlib (only required for the example plotting)
- pandas (for report generation)
- openpyxl (for Excel export)

Install the required packages (recommended in a virtual environment):

```
python -m pip install numpy matplotlib pandas openpyxl
```

## Usage

Run the provided example (from the repository root). The example script is inside the `src` package, so run it on Linux with `PYTHONPATH` set to `src`.

```
env PYTHONPATH=src python -m electrotechnics.examples.example1
```

This will print the Thevenin impedance at a reference frequency and open two plots (magnitude and phase vs frequency).

You can also import the core functions in your code:

```python
# Example 1: Thevenin impedance
from electrotechnics.circuit import z_th_circuit

# edges: list of (n1, n2, z) where z is complex impedance
edges = [(1, 0, 100.0), (1, 2, 1j*2*np.pi*1000*10e-3)]
Z = z_th_circuit(edges, a=1, b=0)

# Example 2: Full circuit solution with voltage and current sources
from electrotechnics.circuit import solve_circuit, report, export_report_to_excel

# Define circuit elements
edges = [
    (1, 0, 100),        # R1 = 100 Ω between nodes 1-0
    (1, 2, 50),         # R2 = 50 Ω between nodes 1-2
    (2, 0, 1j*100)      # XL = j100 Ω between nodes 2-0
]

# Add sources
voltage_sources = [(1, 0, 10+0j)]     # 10 V source between nodes 1-0
current_sources = [(2, 0, 0.1+0j)]    # 0.1 A source between nodes 2-0

# Solve circuit
solution = solve_circuit(edges, voltage_sources, current_sources, ground=0)

# Generate detailed report
report_data = report(edges, voltage_sources, current_sources, solution)

# Access results
node_voltages = solution['node_voltages']           # All node voltages
branch_currents = solution['branch_currents']       # Currents through passive branches
source_currents = solution['voltage_source_currents']  # Currents through voltage sources

# Export report to Excel
export_report_to_excel(report_data, 'circuit_solution.xlsx')

# Example 3: Create a phasor diagram
from electrotechnics.plotting import phasor_plot

# After solving the circuit with solve_circuit(), plot phasors:
plot_out = phasor_plot(
    solution,
    currents=[0, 1],          # Plot currents from branches 0 and 1
    voltages=[(1, 0), (2, 0)],  # Plot voltage differences
    voltage_sources=voltage_sources,
    current_sources=current_sources,
    reference=(1, 0),         # Use V1-0 as angle reference
    separate_scales=True,      # Independent scaling for V/I
    hide_axes=True,
    show_grid=False,
    title='Phasor diagram'
    )
```

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
