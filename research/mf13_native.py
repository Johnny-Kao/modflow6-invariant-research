"""Compare a single EVT segment against an equivalent two-segment curve."""
import argparse
from pathlib import Path
import tempfile
import flopy
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("exe", type=Path)
p.add_argument("phase", choices=["baseline", "corrected"])
args = p.parse_args()

def run_case(ws, nseg, petm0, gh):
    sim = flopy.mf6.MFSimulation(sim_name="evt", sim_ws=ws, exe_name=str(args.exe.resolve()))
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1., 1, 1.)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-10, inner_dvclose=1e-12,
                        outer_maximum=100, inner_maximum=100, linear_acceleration="CG")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=1, delr=1., delc=1., top=20., botm=0.)
    flopy.mf6.ModflowGwfic(gwf, strt=gh)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.)
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=[((0,0,0),gh,10.)])
    row = [(0,0,0),10.,1.,5.]
    if nseg == 2:
        row.extend([0.5,petm0/2])
    row.append(petm0)
    flopy.mf6.ModflowGwfevt(gwf, nseg=nseg, maxbound=1, surf_rate_specified=True,
                          stress_period_data=[tuple(row)], save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbc",
                         saverecord=[("HEAD","ALL"),("BUDGET","ALL")])
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    assert ok, "\n".join(log)
    h = float(flopy.utils.HeadFile(ws/"gwf.hds").get_data().ravel()[0])
    with flopy.utils.CellBudgetFile(ws/"gwf.cbc", precision="double") as b:
        q = float(b.get_data(text="EVT")[-1]["q"][0])
    expected_q = -petm0 * np.clip((h-5.)/5.,0.,1.)
    assert abs(10*(gh-h)+q)<1e-7, (gh,h,q)
    print(f"nseg={nseg} petm0={petm0} ghb={gh} head={h:.12g} evt={q:.12g} oracle={expected_q:.12g}",flush=True)
    return h,q,abs(q-expected_q)<1e-7

with tempfile.TemporaryDirectory(prefix="mf13-") as tmp:
    for petm0 in (0.2,1.0):
        for gh in (4.,7.5,9.9,11.):
            pair = [run_case(Path(tmp)/f"{petm0}-{gh}-{nseg}",nseg,petm0,gh) for nseg in (1,2)]
            assert pair[1][2], pair
            expected = args.phase=="corrected" or petm0==1. or gh<5. or gh>10.1
            assert pair[0][2]==expected, pair
            assert bool(np.allclose(pair[0][:2],pair[1][:2],rtol=0,atol=1e-7))==expected,pair
print("All expected single/two-segment comparisons verified")
