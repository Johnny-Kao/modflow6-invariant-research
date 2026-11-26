"""Test omission of default K22 under rotated three-dimensional anisotropy."""
import argparse
import math
from pathlib import Path
import tempfile
import flopy
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("exe", type=Path)
p.add_argument("phase", choices=["baseline", "corrected"])
args = p.parse_args()

def run_case(ws, angle, explicit, k33):
    sim = flopy.mf6.MFSimulation(sim_name="npf", sim_ws=ws, exe_name=str(args.exe.resolve()))
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.,1.,1.)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-10, inner_dvclose=1e-12,
                        outer_maximum=100, inner_maximum=100, linear_acceleration="CG")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    flopy.mf6.ModflowGwfdis(gwf,nlay=1,nrow=1,ncol=2,delr=1.,delc=1.,top=1.,botm=0.)
    flopy.mf6.ModflowGwfic(gwf,strt=10.)
    flopy.mf6.ModflowGwfnpf(gwf,icelltype=0,k=10.,k33=k33,angle1=0.,angle2=angle,angle3=0.,
                         **({"k22":10.} if explicit else {}))
    flopy.mf6.ModflowGwfchd(gwf,stress_period_data=[((0,0,0),10.)])
    flopy.mf6.ModflowGwfwel(gwf,stress_period_data=[((0,0,1),-1.)])
    flopy.mf6.ModflowGwfoc(gwf,head_filerecord="gwf.hds",budget_filerecord="gwf.cbc",
                          saverecord=[("HEAD","ALL"),("BUDGET","ALL")])
    sim.write_simulation(silent=True)
    ok,log=sim.run_simulation(silent=True,report=True)
    assert ok,"\n".join(log)
    h=float(flopy.utils.HeadFile(ws/"gwf.hds").get_data().ravel()[1])
    theta=math.radians(angle)
    keff=1/(math.cos(theta)**2/10.+math.sin(theta)**2/k33)
    expected=10.-1./keff
    print(f"angle={angle} explicit={explicit} k33={k33} h={h:.12g} oracle={expected:.12g}",flush=True)
    return h,abs(h-expected)<1e-7

with tempfile.TemporaryDirectory(prefix="mf14-") as tmp:
    for k33 in (1.,10.):
        for angle in (0.,45.,90.):
            pair=[run_case(Path(tmp)/f"{k33}-{angle}-{explicit}",angle,explicit,k33) for explicit in (False,True)]
            assert pair[1][1],pair
            expected=args.phase=="corrected" or angle==0. or k33==10.
            assert pair[0][1]==expected,pair
            assert bool(np.isclose(pair[0][0],pair[1][0],rtol=0,atol=1e-7))==expected,pair
print("All expected default-value comparisons verified")
