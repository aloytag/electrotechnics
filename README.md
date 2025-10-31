# Electrotechnics

A small Python library for basic circuit analysis using nodal Laplacian (admittance) matrices.

This repository provides utilities to build complex nodal Laplacian matrices from edge lists and compute Thevenin equivalent impedances between nodes. It includes a small example that computes and plots the Thevenin impedance of a sample circuit as a function of frequency.

## Features

- Build nodal Laplacian (admittance) matrices from edge lists (impedance or admittance input).
- Compute equivalent Thevenin impedance between two nodes using the Moore–Penrose pseudoinverse or test-current method.
- Example script that sweeps frequency and plots magnitude and phase using matplotlib.

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

Install the required packages (recommended in a virtual environment):

```
python -m pip install numpy matplotlib
```

## Usage

Run the provided example (from the repository root). The example script is inside the `src` package, so run it on Linux with `PYTHONPATH` set to `src`.

```
env PYTHONPATH=src python -m electrotechnics.examples.example1
```

This will print the Thevenin impedance at a reference frequency and open two plots (magnitude and phase vs frequency).

You can also import the core function in your code:

```python
from electrotechnics.circuit import z_th_circuit

# edges: list of (n1, n2, z) where z is complex impedance
edges = [(1, 0, 100.0), (1, 2, 1j*2*np.pi*1000*10e-3)]
Z = z_th_circuit(edges, a=1, b=0)
```

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
