"""Native EVT oracle: equivalent one-segment and collinear two-segment curves."""
import argparse
from pathlib import Path
import tempfile

import flopy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=Path)
parser.add_argument("phase", choices=["baseline", "corrected"])
args = parser.parse_args()
exe = str(args.exe.resolve())


def run_case(ws, nseg, petm0):
    targets = [4.0, 7.0, 10.0, 14.9, 16.0]
    sim = flopy.mf6.MFSimulation(sim_name="evt", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, nper=len(targets),
                         perioddata=[(1.0, 1, 1.0)] * len(targets))
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    ims = flopy.mf6.ModflowIms(sim, outer_dvclose=1e-10,
                             inner_dvclose=1e-12, rcloserecord=1e-10,
                             outer_maximum=100, inner_maximum=100,
                             linear_acceleration="BICGSTAB")
    sim.register_ims_package(ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=1,
                           delr=1.0, delc=1.0, top=20.0, botm=0.0)
    flopy.mf6.ModflowGwfic(gwf, strt=targets[0])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.0)
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data={
        i: [((0, 0, 0), head, 100.0)] for i, head in enumerate(targets)
    })
    spd = [(0, 0, 0), 15.0, 1.0, 10.0]
    if nseg == 2:
        spd.extend([0.5, 0.5 * petm0])
    spd.append(petm0)
    flopy.mf6.ModflowGwfevt(gwf, nseg=nseg, maxbound=1,
                           surf_rate_specified=True,
                           stress_period_data=[spd], save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds",
                          budget_filerecord="gwf.cbc",
                          saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    assert ok, "\n".join(log)
    h = flopy.utils.HeadFile(ws / "gwf.hds").get_alldata().ravel()
    with flopy.utils.CellBudgetFile(ws / "gwf.cbc", precision="double") as cb:
        q = np.array([float(record["q"].sum()) for record in cb.get_data(text="EVT")])
        ghb = np.array([float(record["q"].sum()) for record in cb.get_data(text="GHB")])
    assert h.size == q.size == ghb.size == len(targets)
    assert np.max(np.abs(ghb + q)) < 1e-7, (ghb, q)
    expected = -petm0 * np.clip((h - 5.0) / 10.0, 0.0, 1.0)
    matches = np.abs(q - expected) < 1e-8
    expected_matches = np.ones(len(targets), dtype=bool)
    if args.phase == "baseline" and nseg == 1 and petm0 != 1.0:
        expected_matches = (h <= 5.0) | (h >= 15.0)
    for head, actual, oracle, match in zip(h, q, expected, matches):
        print(f"nseg={nseg} petm0={petm0} head={head:.10g} flux={actual:.10g} oracle={oracle:.10g} match={match}", flush=True)
    assert np.array_equal(matches, expected_matches), (h, q, expected, matches)
    return int(matches.sum()), len(matches)


with tempfile.TemporaryDirectory(prefix="evt-oracle-") as tmp:
    count = total = 0
    for nseg in (1, 2):
        for petm0 in (0.1, 0.5, 1.0):
            matched, n = run_case(Path(tmp) / f"{nseg}-{petm0}", nseg, petm0)
            count += matched
            total += n
    print(f"phase={args.phase}: {count}/{total} oracle matches; all expected outcomes verified")
