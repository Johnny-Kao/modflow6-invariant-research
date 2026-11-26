"""SFR accepted-step replay oracle based on upstream test_gwf_ats03.

The disconnected stream uses legal SFR STORAGE inputs; the unchanged nonlinear
aquifer triggers ATS retries independently. No failure injection.
"""
import sys
import tempfile
from pathlib import Path

import flopy
import numpy as np

EXE = str(Path(sys.argv[1]).resolve())
PHASE = sys.argv[2]


def run(ws, steps, mode):
    sim = flopy.mf6.MFSimulation(sim_name="retry", sim_ws=ws, exe_name=EXE)
    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=len(steps),
        perioddata=[(float(dt), 1, 1.0) for dt in steps],
    )
    adaptive = len(steps) == 1
    if adaptive:
        tdis.ats.initialize(maxats=1, perioddata=[(0, 100., 1.e-5, 100., 2., 5.)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    ims = flopy.mf6.ModflowIms(
        sim, outer_dvclose=1.5e-6, outer_maximum=15,
        under_relaxation="DBD", under_relaxation_theta=0.7,
        inner_maximum=5, inner_dvclose=1.5e-6, rcloserecord=1.e-6,
        linear_acceleration="CG", scaling_method="NONE", reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=10,
                           delr=100., delc=1., top=100., botm=0.)
    flopy.mf6.ModflowGwfic(gwf, strt=100.)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=1.)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=0., sy=0.1, transient={0: True})
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={0: [((0, 0, 0), -1.)]})
    chd = flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data={0: [((0, 0, 9), "tshead")]}
    )
    chd.ts.initialize(timeseries=[(0., 100.), (100., 50.)],
                      time_series_namerecord=["tshead"],
                      interpolation_methodrecord=["linearend"])
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sfr = flopy.mf6.ModflowGwfsfr(
        gwf, filename="gwf.sfr", pname="SFR-1",
        length_conversion=1., time_conversion=86400.,
        nreaches=1, packagedata=[(0, (-1, -1, -1), 7200., 10., 1./7200.,
                                   100., 1., 0., .035, 0, 0., 0)],
        connectiondata=[(0,)], perioddata={0: [(0, "inflow", 100.)]},
        stage_filerecord="sfr.stage", budget_filerecord="sfr.cbc",
        print_flows=True,
    )
    sfr.obs.initialize(filename="sfr.obs", digits=18, continuous={
        "sfr.csv": [("stage", "stage", (0,)),
                    ("inflow", "ext-inflow", (0,)),
                    ("outflow", "ext-outflow", (0,))]})
    sim.write_simulation(silent=True)
    # New upstream options may not yet exist in released FloPy's DFN.
    p = ws / "gwf.sfr"
    s = p.read_text()
    assert "BEGIN options" in s
    options = "  STORAGE\n"
    if mode == "tvd":
        options += "  ATS_COURANT 1.0\n"
    s = s.replace("BEGIN options", "BEGIN options\n" + options, 1)
    import re
    s, n = re.subn(r"(?im)^\s*END connectiondata\s*$",
                   "\\g<0>\n\nBEGIN INITIALSTAGES\n  1 100.001\nEND INITIALSTAGES\n", s, count=1)
    assert n == 1
    p.write_text(s)
    ok, buff = sim.run_simulation(silent=True, report=True)
    if not ok:
        raise RuntimeError("\n".join(buff[-40:]))
    listing = (ws / "mfsim.lst").read_text()
    retries = listing.count("will be retried")
    obs = np.atleast_1d(np.genfromtxt(ws / "sfr.csv", names=True, delimiter=","))
    times = obs[obs.dtype.names[0]]
    fields = {k.lower(): obs[k] for k in obs.dtype.names[1:]}
    assert np.isclose(times[-1], 100.)
    return times, fields, retries


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    valid = 0
    divergent = 0
    for mode in ("transient", "tvd"):
        try:
            times, actual, retries = run(root / (mode + "-ats"), [100.], mode)
            print(f"{mode}: retries={retries}; accepted={len(times)}; first dt={times[0]}", flush=True)
            if retries == 0:
                print(f"{mode}: INVALID (no retry)", flush=True)
                continue
            steps = np.diff(times, prepend=0.)
            t_ref, ref, ref_retries = run(root / (mode + "-ref"), steps, mode)
            assert ref_retries == 0 and np.allclose(times, t_ref, rtol=1e-10, atol=1e-9)
            valid += 1
            error = {}
            for k in ("stage", "inflow", "outflow"):
                error[k] = float(np.max(np.abs(actual[k] - ref[k])))
                print(f"{mode} {k}: max_abs_delta={error[k]:.12g}; first ATS={actual[k][0]:.12g}; reference={ref[k][0]:.12g}", flush=True)
            differs = any(not np.allclose(actual[k], ref[k], rtol=1e-8, atol=1e-8)
                          for k in ("stage", "outflow"))
            divergent += int(differs)
            if mode == "tvd":
                for label, obs in [("ats", actual), ("reference", ref)]:
                    depth = obs["stage"] - 100.
                    volumes = depth * 7200. * 10.
                    prev = np.concatenate(([.001 * 7200. * 10.], volumes[:-1]))
                    residual = volumes - prev - steps * (obs["inflow"] + obs["outflow"])
                    print(f"tvd {label}: max accepted-step volume residual={np.max(np.abs(residual)):.12g} m3", flush=True)
        except Exception as exc:
            print(f"{mode}: INVALID {type(exc).__name__}: {exc}", flush=True)
    print(f"{PHASE}: {divergent}/{valid} divergent valid comparisons", flush=True)
    assert valid > 0, "no valid replay"
    assert divergent > 0 if PHASE == "baseline" else divergent == 0
