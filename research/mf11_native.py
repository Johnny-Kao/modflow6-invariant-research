"""NPF default-K22 equivalence on a single structured GWF model."""
import argparse
import math
from pathlib import Path
import tempfile

import flopy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=Path)
parser.add_argument("phase", choices=["baseline", "corrected"])
args = parser.parse_args()
exe = str(args.exe.resolve())


def run_case(ws, angle, k33, explicit, boundary):
    sim = flopy.mf6.MFSimulation(sim_name="mf11", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1,
                         perioddata=[(1., 1, 1.)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-11, inner_dvclose=1e-11,
                        rcloserecord=1e-10, linear_acceleration="BICGSTAB")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=2,
                          delr=1., delc=1., top=1., botm=0.)
    flopy.mf6.ModflowGwfic(gwf, strt=10.)
    kwargs = dict(icelltype=0, k=10., k33=k33,
                  angle1=0., angle2=angle, angle3=0.)
    if explicit:
        kwargs["k22"] = 10.
    flopy.mf6.ModflowGwfnpf(gwf, **kwargs)
    heads = [((0, 0, 0), 10.)]
    if boundary == "fixed_head":
        heads.append(((0, 0, 1), 9.))
    else:
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[((0, 0, 1), -1.)])
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=heads)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds",
                          budget_filerecord="gwf.cbc",
                          saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    success, log = sim.run_simulation(silent=True, report=True)
    assert success, "\n".join(log)
    with flopy.utils.HeadFile(ws / "gwf.hds") as f:
        head = float(f.get_data().ravel()[1])
    with flopy.utils.CellBudgetFile(ws / "gwf.cbc", precision="double") as f:
        flow = f.get_data(text="FLOW-JA-FACE")[-1].ravel()
    assert flow.size == 4, flow
    q = abs(float(flow[1]))
    assert np.isclose(flow[1], -flow[3], atol=1e-9, rtol=0), flow
    phi = math.radians(angle)
    # Orthogonal principal axes, normal along x, unit area and distance.
    k_eff = 1. / (math.cos(phi)**2 / 10. + math.sin(phi)**2 / k33)
    h_expected = 9. if boundary == "fixed_head" else 10. - 1. / k_eff
    q_expected = k_eff if boundary == "fixed_head" else 1.
    ok = bool(np.isclose(head, h_expected, atol=1e-7, rtol=0)
              and np.isclose(q, q_expected, atol=1e-7, rtol=0))
    print(f"angle={angle} k33={k33} explicit_k22={explicit} {boundary} "
          f"head={head:.12g} expected={h_expected:.12g} "
          f"q={q:.12g} expected={q_expected:.12g} match={ok}", flush=True)
    return ok


results = []
with tempfile.TemporaryDirectory(prefix="mf11-") as tmp:
    for angle, k33 in [(0., 1.), (45., 1.), (90., 1.), (45., 10.)]:
        for explicit in [False, True]:
            for boundary in ["fixed_head", "fixed_well"]:
                ok = run_case(Path(tmp) / f"case-{len(results)}",
                              angle, k33, explicit, boundary)
                expected = (args.phase == "corrected" or explicit
                            or angle == 0. or k33 == 10.)
                results.append((ok, expected))
print(f"{args.phase}: {sum(x for x, _ in results)}/{len(results)} oracle matches; "
      f"{sum(x == y for x, y in results)}/{len(results)} expected outcomes")
assert all(x == y for x, y in results), results
