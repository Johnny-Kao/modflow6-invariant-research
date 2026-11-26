"""Native SFR budget volume versus independent cross-section geometry."""
import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import flopy

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=Path)
parser.add_argument("phase", choices=["baseline", "corrected"])
args = parser.parse_args()
exe = str(args.exe.resolve())


def run_case(ws, shape, flow):
    sim = flopy.mf6.MFSimulation(sim_name="mf10", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, time_units="SECONDS", nper=1,
                         perioddata=[(100., 1, 1.)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-9, inner_dvclose=1e-10,
                        linear_acceleration="BICGSTAB")
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=1,
                           delr=100., delc=100., top=10., botm=0.)
    flopy.mf6.ModflowGwfic(gwf, strt=5.)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[((0, 0, 0), 5.)])
    flopy.mf6.ModflowGwfoc(gwf, budget_filerecord="gwf.cbc",
                          saverecord=[("BUDGET", "ALL")])
    extra = {}
    if shape == "trapezoid":
        # 4 m bottom, 10 m bank width, 2 m high, symmetric straight banks.
        flopy.mf6.ModflowUtlsfrtab(gwf, nrow=4, ncol=2,
                                  table=[(0., 2.), (.3, 0.), (.7, 0.), (1., 2.)],
                                  filename="section.tab", pname="section")
        extra["crosssections"] = [(0, "section.tab")]
    flopy.mf6.ModflowGwfsfr(
        gwf, nreaches=1, save_flows=True, length_conversion=1., time_conversion=1.,
        packagedata=[(0, (0, 0, 0), 100., 10., .001, 8., 1., 0., .03, 0, 1., 0)],
        connectiondata=[(0,)], perioddata={0: [(0, "INFLOW", flow)]},
        stage_filerecord="sfr.stage", budget_filerecord="sfr.cbc",
        pname="SFR-1", **extra)
    sim.write_simulation(silent=True)
    success, log = sim.run_simulation(silent=True, report=True)
    assert success, "\n".join(log)
    with flopy.utils.HeadFile(ws / "sfr.stage", text="STAGE") as h:
        depth = float(h.get_data().ravel()[0]) - 8.
    assert 0. < depth < 2., depth
    with flopy.utils.CellBudgetFile(ws / "sfr.cbc", precision="double") as b:
        storage = b.get_data(text="STORAGE")[-1]
        field = next(n for n in storage.dtype.names if n.lower() == "volume")
        actual = float(storage[field][0])
    expected = 100. * (10. * depth if shape == "rectangle"
                       else 4. * depth + 1.5 * depth**2)
    old_formula = 100. * (10. if shape == "rectangle" else 4. + 3. * depth) * depth
    passed = math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8)
    print(f"{shape} Q={flow} depth={depth:.12g} volume={actual:.12g} "
          f"oracle={expected:.12g} old_formula={old_formula:.12g} "
          f"relative_error={actual / expected - 1:.12g}", flush=True)
    if args.phase == "baseline":
        assert math.isclose(actual, old_formula, rel_tol=1e-7, abs_tol=1e-8)
    return passed


results = []
with TemporaryDirectory(prefix="mf10-") as tmp:
    for shape in ("rectangle", "trapezoid"):
        for flow in (1., 3., 5.):
            passed = run_case(Path(tmp) / f"{shape}-{flow}", shape, flow)
            results.append((passed, args.phase == "corrected" or shape == "rectangle"))
print(f"phase={args.phase}: {sum(p for p, _ in results)}/{len(results)} oracle matches")
assert all(p == e for p, e in results), results
