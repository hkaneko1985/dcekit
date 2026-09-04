#!/usr/bin/env python3
"""Physics and materials data generators for extrapolation validation.

The module deliberately keeps the learning representation separate from the
simulators.  All inputs are returned in their original physical units.  The
validation runner applies only componentwise standardization; it does not use
PCA, PLS, or nonlinear feature transforms.

Implemented engines
-------------------
* non-isothermal CSTR: direct integration of mass/energy balances;
* CALPHAD: pycalphad equilibrium with the bundled Pb--Sn TDB;
* lithium-ion battery: PyBaMM SPM with lumped thermal dynamics;
* CO2 adsorption/desorption: IDAES FixedBed1D + PETSc DAE integration;
* methanol synthesis: the official IDAES methanol flowsheet example;
* grain growth: a finite-difference multiphase-field (Allen--Cahn) solver;
* Concrete Slump: the UCI experimental data file.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "physics_validation_results"


@dataclass
class DatasetSpec:
    key: str
    label_ja: str
    engine: str
    engine_status: str
    x_names: list[str]
    x_units: list[str]
    y_names: list[str]
    y_units: list[str]
    full_bounds: np.ndarray
    core_bounds: np.ndarray
    n_core: int
    n_outer: int
    simulator: Callable[[np.ndarray], np.ndarray] | None
    forward_verifiable: bool = True
    notes: str = ""


def lhs(bounds: np.ndarray, n: int, seed: int) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed, optimization="random-cd")
    return qmc.scale(sampler.random(n), bounds[:, 0], bounds[:, 1])


def outer_lhs(
    full_bounds: np.ndarray,
    core_bounds: np.ndarray,
    n: int,
    seed: int,
    buffer_fraction: float = 0.035,
) -> np.ndarray:
    """Latin-hypercube candidates outside a buffered core hyperrectangle."""
    full_bounds = np.asarray(full_bounds, dtype=float)
    core_bounds = np.asarray(core_bounds, dtype=float)
    gap = buffer_fraction * (full_bounds[:, 1] - full_bounds[:, 0])
    lo = core_bounds[:, 0] - gap
    hi = core_bounds[:, 1] + gap
    selected: list[np.ndarray] = []
    batch = max(6 * n, 256)
    iteration = 0
    while sum(len(a) for a in selected) < n:
        candidates = lhs(full_bounds, batch, seed + 1009 * iteration)
        outside = np.any((candidates < lo) | (candidates > hi), axis=1)
        selected.append(candidates[outside])
        iteration += 1
        if iteration > 30:
            raise RuntimeError("Could not sample enough buffered outer points")
    return np.vstack(selected)[:n]


def make_design(spec: DatasetSpec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x_core = lhs(spec.core_bounds, spec.n_core, seed)
    x_outer = outer_lhs(spec.full_bounds, spec.core_bounds, spec.n_outer, seed + 91)
    return x_core, x_outer


# ---------------------------------------------------------------------------
# 1. Non-isothermal CSTR
# ---------------------------------------------------------------------------


def _cstr_one(x: np.ndarray) -> np.ndarray:
    tau, t_feed, t_cool, c_feed, gamma = map(float, x)
    k_ref = 0.18  # min^-1 at T_ref
    t_ref = 350.0  # K
    e_over_r = 7200.0  # K
    heat_rise = 72.0  # K per mol/L reacted
    rho_cp = 4.0  # kJ L^-1 K^-1

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        c_a = max(float(state[0]), 0.0)
        temp = float(np.clip(state[1], 250.0, 700.0))
        k = k_ref * math.exp(-e_over_r * (1.0 / temp - 1.0 / t_ref))
        rate = k * c_a
        dc = (c_feed - c_a) / tau - rate
        dt = (t_feed - temp) / tau + heat_rise * rate - gamma * (temp - t_cool)
        return np.array([dc, dt])

    horizon = max(160.0, 35.0 * tau)
    sol = solve_ivp(
        rhs,
        (0.0, horizon),
        np.array([c_feed, t_feed]),
        method="BDF",
        rtol=2e-8,
        atol=np.array([1e-9, 1e-7]),
    )
    if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
        return np.full(3, np.nan)
    c_a, temp = sol.y[:, -1]
    conversion = np.clip((c_feed - c_a) / max(c_feed, 1e-12), 0.0, 1.0)
    heat_removal = rho_cp * gamma * (temp - t_cool)
    return np.array([conversion, temp, heat_removal])


def simulate_cstr(x: np.ndarray) -> np.ndarray:
    return np.vstack([_cstr_one(row) for row in np.atleast_2d(x)])


# ---------------------------------------------------------------------------
# 2. CALPHAD Pb--Sn equilibrium via pycalphad
# ---------------------------------------------------------------------------


def simulate_calphad(x: np.ndarray) -> np.ndarray:
    import pycalphad
    from pycalphad import Database, equilibrium, variables as v

    calphad_db = Path(pycalphad.__file__).resolve().parent / "tests" / "databases" / "pbsn.tdb"
    if not calphad_db.exists():
        raise FileNotFoundError(
            "The pycalphad test database pbsn.tdb was not found in the installed package."
        )
    db = Database(str(calphad_db))
    phases = ["FCC_A1", "BCT_A5", "LIQUID"]
    rows: list[np.ndarray] = []
    for temp, x_sn in np.atleast_2d(x):
        try:
            eq = equilibrium(
                db,
                ["PB", "SN", "VA"],
                phases,
                {v.P: 101325.0, v.T: float(temp), v.X("SN"): float(x_sn)},
            )
            phase_values = np.asarray(eq.Phase.values).ravel()
            np_values = np.asarray(eq.NP.values, dtype=float).ravel()
            fractions = {p: 0.0 for p in phases}
            for name, amount in zip(phase_values, np_values):
                if isinstance(name, str) and name in fractions and np.isfinite(amount):
                    fractions[name] += float(amount)
            total = sum(fractions.values())
            if total > 0:
                fractions = {k: val / total for k, val in fractions.items()}
            mu_sn = float(np.asarray(eq.MU.sel(component="SN").values).squeeze()) / 1000.0
            rows.append(
                np.array(
                    [fractions["FCC_A1"], fractions["BCT_A5"], fractions["LIQUID"], mu_sn]
                )
            )
        except Exception:
            rows.append(np.full(4, np.nan))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# 3. PyBaMM lithium-ion battery
# ---------------------------------------------------------------------------


def simulate_pybamm(x: np.ndarray) -> np.ndarray:
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    import pybamm

    base = pybamm.ParameterValues("Chen2020")
    lp0 = float(base["Positive electrode thickness [m]"])
    ln0 = float(base["Negative electrode thickness [m]"])
    rp0 = float(base["Positive particle radius [m]"])
    rn0 = float(base["Negative particle radius [m]"])
    rows: list[np.ndarray] = []
    for c_rate, ambient, thickness_scale, radius_scale in np.atleast_2d(x):
        try:
            model = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})
            param = base.copy()
            param.update(
                {
                    "Ambient temperature [K]": float(ambient),
                    "Initial temperature [K]": float(ambient),
                    "Positive electrode thickness [m]": lp0 * float(thickness_scale),
                    "Negative electrode thickness [m]": ln0 * float(thickness_scale),
                    "Positive particle radius [m]": rp0 * float(radius_scale),
                    "Negative particle radius [m]": rn0 * float(radius_scale),
                }
            )
            experiment = pybamm.Experiment([f"Discharge at {float(c_rate):.8g}C until 2.5 V"])
            solver = pybamm.IDAKLUSolver(rtol=2e-6, atol=2e-8)
            sim = pybamm.Simulation(
                model,
                parameter_values=param,
                experiment=experiment,
                solver=solver,
            )
            solution = sim.solve(initial_soc=1.0)
            capacity = float(solution["Discharge capacity [A.h]"].entries[-1])
            time_s = np.asarray(solution["Time [s]"].entries, dtype=float)
            voltage = np.asarray(solution["Terminal voltage [V]"].entries, dtype=float)
            current = np.asarray(solution["Current [A]"].entries, dtype=float)
            energy = float(np.trapezoid(np.maximum(current, 0.0) * voltage, time_s) / 3600.0)
            temperature = np.asarray(
                solution["Volume-averaged cell temperature [K]"].entries, dtype=float
            )
            duration = float(time_s[-1]) / 60.0
            rows.append(np.array([capacity, energy, float(np.max(temperature) - ambient), duration]))
        except Exception:
            rows.append(np.full(4, np.nan))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# 4. IDAES CO2 adsorption/desorption
# ---------------------------------------------------------------------------


def _idaes_fixed_bed_setup(fs, ntfe: int, nxfe: int) -> None:
    from pyomo.environ import TransformationFactory
    from idaes.core import EnergyBalanceType
    from idaes.models_extra.gas_solid_contactors.unit_models.fixed_bed_1D import FixedBed1D
    from idaes_examples.mod.co2_adsorption_desorption.NETL_32D_gas_phase_thermo import (
        GasPhaseParameterBlock,
    )
    from idaes_examples.mod.co2_adsorption_desorption.NETL_32D_solid_phase_thermo import (
        SolidPhaseParameterBlock,
    )
    from idaes_examples.mod.co2_adsorption_desorption.NETL_32D_adsorption_reactions import (
        HeteroReactionParameterBlock,
    )

    fs.gas_properties = GasPhaseParameterBlock()
    fs.solid_properties = SolidPhaseParameterBlock()
    fs.hetero_reactions = HeteroReactionParameterBlock(
        solid_property_package=fs.solid_properties,
        gas_property_package=fs.gas_properties,
    )
    fs.FB = FixedBed1D(
        finite_elements=nxfe,
        transformation_method="dae.finite_difference",
        energy_balance_type=EnergyBalanceType.none,
        pressure_drop_type="ergun_correlation",
        gas_phase_config={"property_package": fs.gas_properties},
        solid_phase_config={
            "property_package": fs.solid_properties,
            "reaction_package": fs.hetero_reactions,
        },
    )
    TransformationFactory("dae.finite_difference").apply_to(
        fs, nfe=ntfe, wrt=fs.time, scheme="BACKWARD"
    )


def _idaes_fix_bed(
    fs,
    gas: dict[str, object],
    solid: dict[str, object] | None,
    bed_diameter: float = 9.0,
    bed_height: float = 1.0,
) -> None:
    blk = fs.FB
    blk.bed_diameter.fix(bed_diameter)
    blk.bed_height.fix(bed_height)
    for t in fs.time:
        blk.gas_inlet.flow_mol[t].fix(gas["flow_mol"])
        blk.gas_inlet.temperature[t].fix(gas["temperature"])
        blk.gas_inlet.pressure[t].fix(gas["pressure"])
        for comp, val in gas["mole_frac_comp"].items():
            blk.gas_inlet.mole_frac_comp[t, comp].fix(val)
    t0 = fs.time.first()
    for xpos in blk.length_domain:
        blk.gas_phase.properties[t0, xpos].flow_mol.fix(gas["flow_mol"])
        blk.gas_phase.properties[t0, xpos].temperature.fix(gas["temperature"])
        for comp, val in gas["mole_frac_comp"].items():
            blk.gas_phase.properties[t0, xpos].mole_frac_comp[comp].fix(val)
        if solid is None:
            blk.solid_properties[t0, xpos].dens_mass_particle.fix()
            blk.solid_properties[t0, xpos].temperature.fix()
            blk.solid_properties[t0, xpos].mass_frac_comp[:].fix()
        else:
            blk.solid_properties[t0, xpos].dens_mass_particle.fix(solid["dens_mass_particle"])
            blk.solid_properties[t0, xpos].temperature.fix(solid["temperature"])
            for comp, val in solid["mass_frac_comp"].items():
                blk.solid_properties[t0, xpos].mass_frac_comp[comp].fix(val)


def _trajectory_mean(tj, variables: list[object], index: int = -1) -> float:
    values = np.array([float(tj.get_vec(var)[index]) for var in variables])
    grid = np.linspace(0.0, 1.0, len(values))
    return float(np.trapezoid(values, grid))


def _simulate_idaes_co2_one(x: np.ndarray, work_root: Path) -> np.ndarray:
    from pyomo.environ import ConcreteModel, Var, units as pyunits
    from idaes.core import FlowsheetBlock
    from idaes.core.solvers import get_solver
    import idaes.core.solvers.petsc as petsc
    from idaes.core.util import scaling as iscale

    t_ads, y_co2, y_h2o, flow_ads, t_des, flow_des = map(float, x)
    y_o2 = 0.12
    y_n2 = 1.0 - y_co2 - y_h2o - y_o2
    if y_n2 <= 0.02:
        return np.full(4, np.nan)

    gas_ads = {
        "flow_mol": flow_ads,
        "temperature": t_ads,
        "pressure": 1.2452e5,
        "mole_frac_comp": {"CO2": y_co2, "H2O": y_h2o, "N2": y_n2, "O2": y_o2},
    }
    solid_ads = {
        "dens_mass_particle": 442.0,
        "temperature": t_ads,
        "mass_frac_comp": {"H2O_s": 1e-8, "Car": 1e-8, "SiO": 1.0},
    }
    gas_des = {
        "flow_mol": flow_des,
        "temperature": t_des,
        "pressure": 1.06525e5,
        "mole_frac_comp": {"CO2": 1e-8, "H2O": 1.0 - 3e-8, "N2": 1e-8, "O2": 1e-8},
    }
    ads_horizon = 1800.0
    des_horizon = 900.0
    nxfe = 5
    solver = get_solver("ipopt")
    solver.options = {
        "max_iter": 120,
        "nlp_scaling_method": "user-scaling",
        "linear_solver": "ma27",
    }
    calc_var_kwds = {"eps": 1e-5}
    ts_options = {
        "--ts_type": "beuler",
        "--ts_dt": 200,
        "--ts_rtol": 20,
        "--ts_save_trajectory": 1,
        "--ksp_rtol": 1e-10,
        "--snes_type": "newtontr",
        "--ts_max_snes_failures": 1000,
    }

    model = ConcreteModel()
    model.fs_ads = FlowsheetBlock(
        dynamic=True,
        time_set=[0.0, ads_horizon / 2.0, ads_horizon],
        time_units=pyunits.s,
    )
    _idaes_fixed_bed_setup(model.fs_ads, ntfe=2, nxfe=nxfe)
    _idaes_fix_bed(model.fs_ads, gas_ads, solid_ads)
    iscale.calculate_scaling_factors(model.fs_ads)
    model.fs_ads.FB.block_triangularization_initialize(
        gas_phase_state_args=gas_ads,
        solid_phase_state_args=solid_ads,
        solver=solver,
        calc_var_kwds=calc_var_kwds,
    )
    model.fs_ads.time_var = Var(model.fs_ads.time)
    model.fs_ads.time_var[0].fix(model.fs_ads.time.first())
    result_ads = petsc.petsc_dae_by_time_element(
        model.fs_ads,
        time=model.fs_ads.time,
        timevar=model.fs_ads.time_var,
        keepfiles=True,
        symbolic_solver_labels=True,
        skip_initial=False,
        ts_options=ts_options,
    )
    tj_ads = result_ads.trajectory
    tf_ads = model.fs_ads.time.last()
    positions = list(model.fs_ads.FB.length_domain)
    car_ads_vars = [
        model.fs_ads.FB.solid_properties[tf_ads, xpos].mass_frac_comp["Car"]
        for xpos in positions
    ]
    car_ads = _trajectory_mean(tj_ads, car_ads_vars)
    outlet_ratio = float(
        tj_ads.get_vec(model.fs_ads.FB.gas_outlet.mole_frac_comp[tf_ads, "CO2"])[-1]
        / y_co2
    )

    model.fs_des = FlowsheetBlock(
        dynamic=True,
        time_set=[0.0, des_horizon / 2.0, des_horizon],
        time_units=pyunits.s,
    )
    _idaes_fixed_bed_setup(model.fs_des, ntfe=2, nxfe=nxfe)
    tf_ads_idx = -1
    components = model.fs_des.FB.config.solid_phase_config.property_package.component_list
    for t in model.fs_des.time:
        for xpos in model.fs_des.FB.length_domain:
            model.fs_des.FB.solid_properties[t, xpos].temperature.set_value(t_des)
            model.fs_des.FB.solid_properties[t, xpos].dens_mass_particle.set_value(
                tj_ads.get_vec(
                    model.fs_ads.FB.solid_properties[tf_ads, xpos].dens_mass_particle
                )[tf_ads_idx]
            )
            for comp in components:
                model.fs_des.FB.solid_properties[t, xpos].mass_frac_comp[comp].set_value(
                    tj_ads.get_vec(
                        model.fs_ads.FB.solid_properties[tf_ads, xpos].mass_frac_comp[comp]
                    )[tf_ads_idx]
                )
    _idaes_fix_bed(model.fs_des, gas_des, None)
    iscale.calculate_scaling_factors(model.fs_des)
    model.fs_des.FB.block_triangularization_initialize(
        gas_phase_state_args=gas_des,
        solver=solver,
        calc_var_kwds=calc_var_kwds,
    )
    model.fs_des.time_var = Var(model.fs_des.time)
    model.fs_des.time_var[0].fix(model.fs_des.time.first())
    result_des = petsc.petsc_dae_by_time_element(
        model.fs_des,
        time=model.fs_des.time,
        timevar=model.fs_des.time_var,
        keepfiles=True,
        symbolic_solver_labels=True,
        skip_initial=False,
        ts_options=ts_options,
    )
    tj_des = result_des.trajectory
    tf_des = model.fs_des.time.last()
    car_des_vars = [
        model.fs_des.FB.solid_properties[tf_des, xpos].mass_frac_comp["Car"]
        for xpos in positions
    ]
    car_des = _trajectory_mean(tj_des, car_des_vars)
    recovery = float(np.clip((car_ads - car_des) / max(car_ads, 1e-12), -0.2, 1.2))
    des_times = np.asarray(tj_des.time, dtype=float)
    des_y = np.asarray(
        tj_des.get_vec(model.fs_des.FB.gas_outlet.mole_frac_comp[tf_des, "CO2"]),
        dtype=float,
    )
    des_h2o = np.asarray(
        tj_des.get_vec(model.fs_des.FB.gas_outlet.mole_frac_comp[tf_des, "H2O"]),
        dtype=float,
    )
    dry_y = des_y / np.maximum(1.0 - des_h2o, 1e-10)
    purity = float(
        np.trapezoid(np.clip(dry_y, 0.0, 1.0), des_times)
        / max(des_times[-1] - des_times[0], 1e-12)
    )
    return np.array([car_ads, recovery, purity, outlet_ratio])


def simulate_idaes_co2(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    rows: list[np.ndarray] = []
    work_root = RESULT_ROOT / "idaes_co2_work"
    work_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(x):
        try:
            with tempfile.TemporaryDirectory(prefix=f"case_{index:04d}_", dir=work_root) as tmp:
                previous = Path.cwd()
                try:
                    os.chdir(tmp)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        result = _simulate_idaes_co2_one(row, work_root)
                finally:
                    os.chdir(previous)
            rows.append(result)
        except Exception as exc:
            warnings.warn(f"IDAES CO2 case failed: {exc}")
            rows.append(np.full(4, np.nan))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# 5. IDAES methanol synthesis flowsheet
# ---------------------------------------------------------------------------


class IDAESMethanolBatch:
    """Build the official IDAES example once and solve a continuation batch."""

    def __init__(self) -> None:
        from pyomo.environ import ConcreteModel
        from idaes.core.solvers import get_solver
        from idaes_examples.notebooks.docs.flowsheets import methanol_flowsheet as mf

        import idaes.logger as idaeslog

        idaeslog.getLogger("idaes").setLevel(logging.ERROR)
        self.mf = mf
        self.model = ConcreteModel()
        self.solver = get_solver()
        self.solver.options = {"tol": 1e-7, "max_iter": 500, "linear_solver": "ma27"}
        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                mf.build_model(self.model)
                mf.set_inputs(self.model)
                mf.scale_flowsheet(self.model)
                mf.initialize_flowsheet(self.model)
                self.solver.solve(self.model, tee=False)
        finally:
            logging.disable(previous_disable)

    def solve_one(self, x: np.ndarray) -> np.ndarray:
        from pyomo.environ import value

        feed_scale, ratio, pressure_bar, preheat, conversion, flash_temp = map(float, x)
        model = self.model
        total = 954.0 * feed_scale
        flow_co = total / (1.0 + ratio)
        flow_h2 = ratio * flow_co
        model.fs.H2.outlet.flow_mol[0].fix(flow_h2)
        model.fs.CO.outlet.flow_mol[0].fix(flow_co)
        model.fs.C101.outlet.pressure.fix(pressure_bar * 1e5)
        model.fs.H101.outlet_temp.set_value(
            model.fs.H101.control_volume.properties_out[0].temperature == preheat
        )
        model.fs.R101.conversion.fix(conversion)
        model.fs.H102.outlet_temp.set_value(
            model.fs.H102.control_volume.properties_out[0].temperature == flash_temp
        )
        model.fs.F101.outlet_temp.set_value(
            model.fs.F101.control_volume.properties_out[0].temperature == flash_temp
        )
        result = self.solver.solve(model, tee=False)
        if str(result.solver.termination_condition).lower() != "optimal":
            return np.full(4, np.nan)
        product = value(
            model.fs.CH3OH.inlet.flow_mol[0]
            * model.fs.CH3OH.inlet.mole_frac_comp[0, "CH3OH"]
        )
        recovery = value(model.fs.F101.recovery)
        reactor_heat = -value(model.fs.R101.heat_duty[0]) / 1e6
        utility = (
            abs(value(model.fs.C101.work_mechanical[0]))
            + abs(value(model.fs.H101.heat_duty[0]))
            + abs(value(model.fs.R101.heat_duty[0]))
            + abs(value(model.fs.H102.heat_duty[0]))
            + abs(value(model.fs.F101.heat_duty[0]))
            - abs(value(model.fs.T101.work_isentropic[0]))
        ) / 1e6
        return np.array([product, recovery, reactor_heat, utility])

    def solve_continuation(self, target: np.ndarray, steps: int = 6) -> np.ndarray:
        """Move from the documented base case to a remote specification."""
        base = np.array([1.0, 637.2 / 316.8, 51.0, 488.15, 0.75, 407.15])
        result = np.full(4, np.nan)
        for fraction in np.linspace(1.0 / steps, 1.0, steps):
            point = base + fraction * (np.asarray(target, dtype=float) - base)
            result = self.solve_one(point)
            if not np.all(np.isfinite(result)):
                break
        return result

    def solve(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for row in np.atleast_2d(x):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = self.solve_one(row)
                if not np.all(np.isfinite(result)):
                    fresh = IDAESMethanolBatch()
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        result = fresh.solve_continuation(row)
                    if np.all(np.isfinite(result)):
                        self.model = fresh.model
                        self.solver = fresh.solver
                rows.append(result)
            except Exception:
                rows.append(np.full(4, np.nan))
        return np.vstack(rows)


_METHANOL_BATCH: IDAESMethanolBatch | None = None


def simulate_idaes_methanol(x: np.ndarray) -> np.ndarray:
    global _METHANOL_BATCH
    if _METHANOL_BATCH is None:
        _METHANOL_BATCH = IDAESMethanolBatch()
    return _METHANOL_BATCH.solve(x)


# ---------------------------------------------------------------------------
# 6. Multiphase-field grain growth
# ---------------------------------------------------------------------------


_NUCLEI_RNG = np.random.default_rng(20260823)
_NUCLEI = _NUCLEI_RNG.uniform(0.0, 32.0, size=(40, 2))


def _phase_field_one(x: np.ndarray) -> np.ndarray:
    temp, anneal_time, mobility_prefactor, kappa, nuclei_value = map(float, x)
    n_grains = int(np.clip(np.rint(nuclei_value), 8, 40))
    size = 32
    yy, xx = np.mgrid[0:size, 0:size]
    dx = np.abs(xx[None, :, :] - _NUCLEI[:n_grains, 0, None, None])
    dy = np.abs(yy[None, :, :] - _NUCLEI[:n_grains, 1, None, None])
    dx = np.minimum(dx, size - dx)
    dy = np.minimum(dy, size - dy)
    labels0 = np.argmin(dx * dx + dy * dy, axis=0)
    eta = np.zeros((n_grains, size, size), dtype=np.float64)
    for grain in range(n_grains):
        eta[grain] = gaussian_filter((labels0 == grain).astype(float), sigma=0.65, mode="wrap")
    eta /= np.maximum(np.sum(eta, axis=0, keepdims=True), 1e-12)

    q_over_r = 4200.0
    mobility = mobility_prefactor * math.exp(-q_over_r * (1.0 / temp - 1.0 / 1000.0))
    gamma_cross = 1.45
    n_steps = int(np.clip(round(110.0 * anneal_time), 45, 420))
    dt = min(0.030, 0.055 / max(mobility * kappa, 1.0))
    for _ in range(n_steps):
        lap = (
            np.roll(eta, 1, axis=1)
            + np.roll(eta, -1, axis=1)
            + np.roll(eta, 1, axis=2)
            + np.roll(eta, -1, axis=2)
            - 4.0 * eta
        )
        sum_sq = np.sum(eta * eta, axis=0, keepdims=True)
        derivative = eta**3 - eta + 2.0 * gamma_cross * eta * (sum_sq - eta * eta) - kappa * lap
        eta -= dt * mobility * derivative
        eta = np.clip(eta, -0.05, 1.10)

    labels = np.argmax(eta, axis=0)
    areas = np.bincount(labels.ravel(), minlength=n_grains).astype(float)
    active = areas >= 4.0
    active_areas = areas[active]
    diameter = np.sqrt(4.0 * active_areas / np.pi)
    boundary = 0.5 * (
        np.mean(labels != np.roll(labels, 1, axis=0))
        + np.mean(labels != np.roll(labels, 1, axis=1))
    )
    mean_diameter = float(np.mean(diameter))
    grain_count = float(len(active_areas))
    size_cv = float(np.std(diameter, ddof=1) / max(mean_diameter, 1e-12)) if len(diameter) > 1 else 0.0
    return np.array([mean_diameter, grain_count, float(boundary), size_cv])


def simulate_phase_field(x: np.ndarray) -> np.ndarray:
    return np.vstack([_phase_field_one(row) for row in np.atleast_2d(x)])


# ---------------------------------------------------------------------------
# 7. UCI Concrete Slump experimental data
# ---------------------------------------------------------------------------


def load_concrete_slump(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = path or ROOT / "data" / "raw" / "slump_test.data"
    frame = pd.read_csv(path)
    x = frame[
        ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]
    ].to_numpy(dtype=float)
    y = frame[["SLUMP(cm)", "FLOW(cm)", "Compressive Strength (28-day)(Mpa)"]].to_numpy(
        dtype=float
    )
    return x, y


def concrete_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unsupervised inner/outer split based only on standardized ingredients."""
    from sklearn.preprocessing import StandardScaler

    x, y = load_concrete_slump()
    z = StandardScaler().fit_transform(x)
    radius = np.linalg.norm(z, axis=1)
    order = np.argsort(radius)
    n = len(x)
    n_train = int(round(0.70 * n))
    n_buffer = int(round(0.10 * n))
    train_idx = order[:n_train]
    test_idx = order[n_train + n_buffer :]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            "D1",
            "Non-isothermal CSTR",
            "SciPy BDF integration of material and energy balances",
            "Equation implementation",
            ["residence_time", "feed_temperature", "coolant_temperature", "feed_concentration", "heat_transfer_rate"],
            ["min", "K", "K", "mol/L", "1/min"],
            ["conversion", "reactor_temperature", "heat_removal"],
            ["-", "K", "kJ/(L min)"],
            np.array([[0.9, 8.5], [326, 362], [288, 332], [0.55, 1.45], [0.08, 0.75]]),
            np.array([[2.0, 6.2], [334, 353], [299, 322], [0.78, 1.22], [0.20, 0.56]]),
            170,
            85,
            simulate_cstr,
            notes="First-order A-to-B reaction with Arrhenius kinetics and heat removal.",
        ),
        DatasetSpec(
            "D2",
            "CALPHAD Pb-Sn phase equilibrium",
            "pycalphad 0.11.2 + bundled pbsn.tdb",
            "Public library",
            ["temperature", "x_sn"],
            ["K", "mol/mol"],
            ["fcc_fraction", "bct_fraction", "liquid_fraction", "mu_sn"],
            ["-", "-", "-", "kJ/mol"],
            np.array([[300, 620], [0.02, 0.98]]),
            np.array([[365, 525], [0.14, 0.86]]),
            150,
            80,
            simulate_calphad,
            notes="Equilibrium phase fractions of FCC_A1, BCT_A5, and LIQUID at 101325 Pa.",
        ),
        DatasetSpec(
            "D3",
            "PyBaMM lithium-ion battery",
            "PyBaMM 26.8 SPM + lumped thermal + Chen2020",
            "Public library",
            ["c_rate", "ambient_temperature", "electrode_thickness_scale", "particle_radius_scale"],
            ["C", "K", "-", "-"],
            ["discharge_capacity", "discharge_energy", "peak_temperature_rise", "discharge_time"],
            ["Ah", "Wh", "K", "min"],
            np.array([[0.25, 2.8], [274, 322], [0.76, 1.24], [0.72, 1.28]]),
            np.array([[0.55, 1.55], [288, 308], [0.90, 1.10], [0.90, 1.10]]),
            125,
            65,
            simulate_pybamm,
            notes="Constant-current discharge from full charge to 2.5 V.",
        ),
        DatasetSpec(
            "D4",
            "IDAES carbon-dioxide adsorption/desorption",
            "IDAES FixedBed1D + NETL_32D + PETSc DAE",
            "Public library with reduced space/time discretization",
            ["adsorption_temperature", "co2_feed_fraction", "h2o_feed_fraction", "adsorption_flow", "desorption_temperature", "desorption_flow"],
            ["K", "-", "-", "mol/s", "K", "mol/s"],
            ["carbamate_loading", "regeneration_recovery", "desorption_co2_purity", "breakthrough_ratio"],
            ["kg/kg", "-", "-", "-"],
            np.array([[296, 314], [0.025, 0.110], [0.045, 0.120], [2.2, 4.0], [430, 490], [6.0, 13.0]]),
            np.array([[300, 308], [0.040, 0.075], [0.070, 0.100], [2.7, 3.5], [450, 475], [8.0, 11.0]]),
            34,
            18,
            simulate_idaes_co2,
            notes="Five spatial elements; 1800 s adsorption and 900 s desorption; no energy balance.",
        ),
        DatasetSpec(
            "D5",
            "IDAES methanol-synthesis process",
            "IDAES official methanol flowsheet + IPOPT",
            "Public library",
            ["feed_scale", "h2_co_ratio", "compressor_pressure", "preheat_temperature", "reactor_conversion", "flash_temperature"],
            ["-", "mol/mol", "bar", "K", "-", "K"],
            ["methanol_product", "methanol_recovery", "reactor_cooling", "total_utility"],
            ["mol/s", "-", "MW", "MW"],
            np.array([[0.78, 1.22], [1.70, 2.35], [45, 65], [465, 505], [0.65, 0.86], [385, 425]]),
            np.array([[0.90, 1.10], [1.88, 2.15], [50, 59], [480, 496], [0.71, 0.81], [397, 414]]),
            125,
            65,
            simulate_idaes_methanol,
            notes="Steady-state flowsheet with reactor conversion used as a design specification.",
        ),
        DatasetSpec(
            "D6",
            "Multiphase-field grain growth",
            "2D multiphase Allen–Cahn finite difference",
            "Equation implementation",
            ["temperature", "anneal_time", "mobility_prefactor", "gradient_coefficient", "initial_grain_count"],
            ["K", "a.u.", "-", "-", "count"],
            ["mean_grain_diameter", "grain_count", "boundary_density", "grain_size_cv"],
            ["pixel", "count", "-", "-"],
            np.array([[780, 1220], [0.45, 3.2], [0.55, 1.45], [0.70, 1.30], [10, 32]]),
            np.array([[900, 1100], [1.0, 2.4], [0.78, 1.22], [0.84, 1.16], [15, 27]]),
            145,
            75,
            simulate_phase_field,
            notes="32-by-32 periodic grid with fixed nuclei; temperature controls Arrhenius boundary mobility.",
        ),
        DatasetSpec(
            "D7",
            "Concrete Slump experimental data",
            "UCI Concrete Slump Test (DOI 10.24432/C5FG7D)",
            "Experimental data",
            ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate"],
            ["kg/m3"] * 7,
            ["slump", "flow", "compressive_strength_28d"],
            ["cm", "cm", "MPa"],
            np.zeros((7, 2)),
            np.zeros((7, 2)),
            72,
            21,
            None,
            forward_verifiable=False,
            notes="103 mixtures; the inner 70% by an input-only standardized radius is training data and the outer 20% is test data.",
        ),
    ]


def generate_dataset(
    spec: DatasetSpec,
    seed: int = 20260823,
    failure_log_path: Path | None = None,
) -> pd.DataFrame:
    if spec.key == "D7":
        x_train, y_train, x_test, y_test = concrete_split()
    else:
        if spec.simulator is None:
            raise ValueError(f"No simulator for {spec.key}")
        x_train, x_test = make_design(spec, seed + int(spec.key[1:]) * 100)
        y_train = spec.simulator(x_train)
        y_test = spec.simulator(x_test)
    columns = spec.x_names + spec.y_names
    train = pd.DataFrame(np.column_stack([x_train, y_train]), columns=columns)
    train.insert(0, "split", "train")
    test = pd.DataFrame(np.column_stack([x_test, y_test]), columns=columns)
    test.insert(0, "split", "test")
    frame = pd.concat([train, test], ignore_index=True)
    finite = np.all(np.isfinite(frame[spec.y_names].to_numpy(dtype=float)), axis=1)
    if not np.all(finite):
        warnings.warn(f"{spec.key}: dropping {np.sum(~finite)} failed simulations")
        if failure_log_path is not None:
            failure_log_path.parent.mkdir(parents=True, exist_ok=True)
            failed = frame.loc[~finite].copy()
            failed.insert(1, "simulation_status", "failed_or_nonfinite")
            failed.to_csv(failure_log_path, index=False)
        frame = frame.loc[finite].reset_index(drop=True)
    elif failure_log_path is not None:
        empty = frame.iloc[0:0].copy()
        empty.insert(1, "simulation_status", pd.Series(dtype=str))
        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(failure_log_path, index=False)
    return frame


def configure_runtime_environment() -> None:
    workspace = str(ROOT)
    os.environ.setdefault("IDAES_DATA", str(ROOT / ".idaes"))
    os.environ.setdefault("PYOMO_CONFIG_DIR", str(ROOT / ".pyomo"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    idaes_bin = str(ROOT / ".idaes" / "bin")
    os.environ["PATH"] = idaes_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = idaes_bin + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    for logger_name in ("idaes", "pyomo", "pyomo.core", "pyomo.solvers"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    Path(os.environ["IDAES_DATA"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["PYOMO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    _ = workspace


configure_runtime_environment()
