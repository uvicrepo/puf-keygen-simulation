# ece548_gradProject

Reed-Solomon Error Correction for PUF Key Generation
Simulation comparing RS performance under independent vs correlated bit errors.


## Requirements ##

Python 3.8 or newer
Libraries: reedsolo, numpy, matplotlib


## Setup ##

Linux / Mac:
    python3 -m venv venv
    source venv/bin/activate
    pip install reedsolo numpy matplotlib

Windows:
    python -m venv venv
    venv\Scripts\activate
    pip install reedsolo numpy matplotlib


## Running the simulation ##

Linux / Mac:
    source venv/bin/activate
    python puf_rs_simulation.py

Windows:
    venv\Scripts\activate
    python puf_rs_simulation.py


## Files ##

puf_rs_simulation.py       Main simulation script (only file needed to run)
fig1_failure_rates.png     Failure rate comparison (independent vs correlated)
fig2_symbol_errors.png     Symbol error rate vs correction capacity
fig3_efficiency_gain.png   RS efficiency gain from correlated errors

PDF versions of each plot can also be generated.
