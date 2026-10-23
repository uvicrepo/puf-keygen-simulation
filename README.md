# PUF Key Generation with Reed-Solomon Error Correction

This project simulates Reed-Solomon (RS) error correction for Physically Unclonable Function (PUF) key generation under different error models.

The simulation compares performance when bit errors are:
- independent
- correlated

The goal is to evaluate how Reed-Solomon coding improves reliability and robustness for PUF-based key generation.

## Overview

PUF-derived keys are susceptible to noise and environmental variation. Reed-Solomon codes can help recover reliable keys by correcting symbol errors introduced during extraction and transmission.

This project compares:
- failure rates under independent bit errors
- failure rates under correlated bit errors
- symbol error rate as a function of correction capacity
- efficiency gains achieved by exploiting correlated error behavior

## Requirements

- Python 3.8 or newer
- numpy
- matplotlib
- reedsolo

## Installation

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install reedsolo numpy matplotlib
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install reedsolo numpy matplotlib
```

## Run the Simulation

### Linux / macOS

```bash
source venv/bin/activate
python puf_rs_simulation.py
```

### Windows

```bash
venv\Scripts\activate
python puf_rs_simulation.py
```

## Project Files

- `puf_rs_simulation.py` — main simulation script
- `fig1_failure_rates.png` — failure rate comparison under independent and correlated errors
- `fig2_symbol_errors.png` — symbol error rate versus correction capacity
- `fig3_efficiency_gain.png` — Reed-Solomon efficiency gain from correlated errors

PDF versions of the generated plots can also be produced.

## Notes

This project is intended as a numerical and visual analysis of Reed-Solomon error-correction performance in a PUF key generation context. It is designed to support evaluation of how error correlation affects coding efficiency and key reliability.
