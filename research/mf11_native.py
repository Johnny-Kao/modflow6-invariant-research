"""Check storage conservation across a repeated groundwater time step."""
import argparse
from pathlib import Path
import re
import tempfile

import flopy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=Path)
parser.add_argument("phase", choices=["baseline", "corrected"])
args = parser.parse_args()
exe = str(args.exe.resolve())


def run_case(ws, retry):
    sim = flopy.mf6.MFSimulation(sim_name="tvs", sim_ws=ws, exe_name=exe)
    tdis = flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(2.0, 2, 1.0)])
    if retry:
        tdis.ats.initialize(maxats=1, perioddata=[(0, 1.0, 1e-5, 1.0, 1.0, 5.0)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    ims = flopy.mf6.ModflowIms(
        sim, outer_dvclose=2.0 if retry else 1e-10,
        outer_maximum=1 if retry else 100, inner_dvclose=1e-12,
        inner_maximum=100, rcloserecord=1e-10, linear_acceleration="CG",
    )
    sim.register_ims_package(ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=1, delr=1.0, delc=1.0, top=20.0, botm=0.0,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=10.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.0)
    sto = flopy.mf6.ModflowGwfsto(
        gwf, iconvert=0, ss=0.1, transient={0: True},
    )
    tvs = flopy.mf6.ModflowUtltvs(
        sto, print_input=True, perioddata={0: [((0, 0, 0), "SS", "ssts")]},
        filename="gwf.tvs",
    )
    tvs.ts.initialize(
        filename="gwf.tvs.ts",
        timeseries=[(0.0, 0.1), (1.0, 0.1), (2.0, 0.2)],
        time_series_namerecord="ssts", interpolation_methodrecord="linearend",
    )
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    assert ok, "\n".join(log)
    listing = (ws / "mfsim.lst").read_text()
    attempts = re.findall(r"step\s+2\s+and\s+period\s+1\s+will be retried", listing, re.I)
    if retry:
        assert attempts, "no retry of second step observed"
    hfile = flopy.utils.HeadFile(ws / "gwf.hds")
    t = np.array(hfile.get_times())
    h = hfile.get_alldata().ravel()
    assert abs(t[0] - 1.0) < 1e-9 and abs(t[-1] - 2.0) < 1e-9, t
    ss = np.interp(t, [0.0, 1.0, 2.0], [0.1, 0.1, 0.2])
    # Confined cell, no sources/sinks: SS times volume times head is invariant
    # for this storage formulation, including time-varying properties.
    inventory = 20.0 * ss * h
    error = np.max(np.abs(inventory - 20.0))
    print(f"retry={retry} second-step-retries={len(attempts)} max-inventory-error={error:.12g}", flush=True)
    print(f"first 8 t={t[:8]} h={h[:8]} inventory={inventory[:8]}; final inventory={inventory[-1]}", flush=True)
    return error < 1e-7


with tempfile.TemporaryDirectory(prefix="mf11-") as tmp:
    control = run_case(Path(tmp) / "control", False)
    assert control, "fixed-step control failed"
    retry = run_case(Path(tmp) / "retry", True)
    assert retry == (args.phase == "corrected"), (args.phase, retry)
