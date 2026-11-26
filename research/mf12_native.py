"""Check the documented wetting direction with a two-layer GWF column."""
import argparse
from pathlib import Path
import tempfile

import flopy

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=Path)
parser.add_argument("phase", choices=["baseline", "corrected"])
args = parser.parse_args()
exe = str(args.exe.resolve())


def run_case(ws, source_layer, wetdry):
    target = 1 - source_layer
    sim = flopy.mf6.MFSimulation(sim_name="rewet", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    ims = flopy.mf6.ModflowIms(
        sim, outer_dvclose=1e-9, outer_maximum=100,
        inner_dvclose=1e-11, inner_maximum=100, rcloserecord=1e-9,
        linear_acceleration="CG",
    )
    sim.register_ims_package(ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=2, nrow=1, ncol=1, delr=1.0, delc=1.0,
        top=20.0, botm=[10.0, 0.0],
    )
    heads = [15.0, 15.0]
    heads[target] = [9.0, -1.0][target]
    thresholds = [0.0, 0.0]
    thresholds[target] = wetdry
    flopy.mf6.ModflowGwfic(gwf, strt=heads)
    flopy.mf6.ModflowGwfnpf(
        gwf, icelltype=1, k=1.0, wetdry=thresholds,
        rewet_record=[("WETFCT", 1.0, "IWETIT", 1, "IHDWET", 1)],
    )
    flopy.mf6.ModflowGwfghb(
        gwf, stress_period_data=[((source_layer, 0, 0), 15.0, 1.0)],
    )
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    assert ok, "\n".join(log)
    h = flopy.utils.HeadFile(ws / "gwf.hds").get_data().ravel()
    assert abs(h[source_layer] - 15.0) < 1e-6, h
    active = h[target] > -1e20
    if active:
        assert abs(h[target] - 15.0) < 1e-6, h
    expected = source_layer == 1 and wetdry != 0.0
    print(f"source_layer={source_layer} WETDRY={wetdry} heads={h} active={active} expected={expected}", flush=True)
    return bool(active) == expected


with tempfile.TemporaryDirectory(prefix="mf12-") as root:
    results = []
    for source in (0, 1):
        for wetdry in (-0.1, 0.1, 0.0):
            passed = run_case(Path(root) / f"{source}-{wetdry}", source, wetdry)
            expected_pass = args.phase == "corrected" or source == 1 or wetdry == 0.0
            results.append((passed, expected_pass))
    print(f"{args.phase}: {sum(p for p, _ in results)}/{len(results)} documented outcomes; {sum(p == e for p, e in results)}/{len(results)} expected results", flush=True)
    assert all(p == e for p, e in results), results
