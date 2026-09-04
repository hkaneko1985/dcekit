# Optional physical-simulator environment

The cached D1-D7 datasets are sufficient for the statistical reanalysis,
strong-baseline comparison, directional diagnostics, and Bayesian optimization.
Regenerating D2-D5 or verifying newly generated inverse candidates requires the
optional packages pinned in `requirements-physics.txt`.

## External requirements

- D2: pycalphad 0.11.2 and the bundled `pbsn.tdb` database.
- D3: PyBaMM 26.8.0.0 with the Chen2020 parameter set.
- D4: IDAES-PSE 2.12.0, Pyomo 6.10.1, PETSc support, and the solver stack used
  by the IDAES FixedBed1D example.
- D5: IDAES-PSE 2.12.0, IDAES Examples 2.10.0, Pyomo 6.10.1, and IPOPT.

Install the Python layer with:

```bash
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-physics.txt
```

IDAES and PETSc/IPOPT binaries remain platform-specific. Record their exact
binary versions and availability in the generated `run_manifest.json`. A
regenerated analysis must also retain `D*_failed_simulations.csv`; failed or
nonfinite simulator inputs must not be silently discarded.

The archived D4 calculations used five spatial elements and two finite time
intervals for each adsorption/desorption stage. These settings are intended for
method validation, not a converged process-design calculation. Any physical
interpretation of D4 should be accompanied by a mesh/time-step sensitivity
study. D6 similarly uses a fixed 32-by-32 grid and should not be treated as a
grid-converged microstructure simulation.
