"""Compare equivalent NPF inputs against an independent principal-axis oracle."""
import argparse
import math
from pathlib import Path
import tempfile

import flopy

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=Path)
parser.add_argument("phase", choices=["baseline", "corrected"])
args = parser.parse_args()
exe = str(args.exe.resolve())


def run_case(ws, axis, tilt, explicit, k33):
    sim = flopy.mf6.MFSimulation(sim_name="npf", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1., 1, 1.)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-10, inner_dvclose=1e-12,
                         rcloserecord=1e-10, linear_acceleration="CG")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    shape = (1, 2) if axis == "x" else (2, 1)
    target = (0, 0, 1) if axis == "x" else (0, 1, 0)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=shape[0], ncol=shape[1],
                            delr=1., delc=1., top=1., botm=0.)
    flopy.mf6.ModflowGwfic(gwf, strt=20.)
    options = dict(icelltype=0, k=10., k33=k33, angle1=0.,
                   angle2=tilt, angle3=0.)
    if explicit:
        options["k22"] = 10.
    flopy.mf6.ModflowGwfnpf(gwf, **options)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[((0, 0, 0), 20.)])
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[(target, -1.)])
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds",
                           budget_filerecord="gwf.cbc",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    assert ok, "\n".join(log)
    h = float(flopy.utils.HeadFile(ws / "gwf.hds").get_data()[target])
    # Unit face width, thickness and total connection length; harmonic
    # directional conductivity of a tensor rotated about its y axis.
    t = math.radians(tilt)
    inv_k = (math.cos(t)**2 / 10. + math.sin(t)**2 / k33
             if axis == "x" else 0.1)
    expected = 20. - inv_k
    matches = abs(h - expected) < 1e-7
    print(f"axis={axis} tilt={tilt} explicit_k22={explicit} k33={k33:g} "
          f"head={h:.12g} oracle={expected:.12g} inferred_K={1/(20.-h):.12g} "
          f"matches={matches}", flush=True)
    return matches


results = []
with tempfile.TemporaryDirectory(prefix="npf-k22-") as tmp:
    for axis, tilt, k33 in [("x", 0., 1.), ("x", 45., 1.),
                            ("x", 90., 1.), ("y", 90., 1.),
                            ("x", 90., 10.)]:
        for explicit in (False, True):
            matches = run_case(Path(tmp) / str(len(results)), axis, tilt, explicit, k33)
            expected_pass = (args.phase == "corrected" or explicit or
                             axis == "y" or tilt == 0. or k33 == 10.)
            results.append((matches, expected_pass))
print(f"{args.phase}: {sum(x for x, _ in results)}/{len(results)} oracle matches")
assert all(x == y for x, y in results), results
