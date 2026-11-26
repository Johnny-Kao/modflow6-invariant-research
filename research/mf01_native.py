"""Native two-model head/flow oracle for rotated anisotropy across an exchange."""
import argparse
import math
import pathlib
import tempfile
import flopy

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=pathlib.Path)
parser.add_argument("phase", choices=["baseline", "corrected"])
args = parser.parse_args()
exe = str(args.exe.resolve())

def run_case(workspace, angle, reverse, k1, k2):
    theta = math.radians(angle)
    u = (math.cos(theta), math.sin(theta))
    v = (-u[1], u[0])
    sim = flopy.mf6.MFSimulation(sim_name="mf01", sim_ws=workspace, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1., 1, 1.)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-10, inner_dvclose=1e-10,
                         rcloserecord=1e-10, outer_maximum=100, inner_maximum=100,
                         linear_acceleration="BICGSTAB")
    for name, center in [("a", (0., 0.)), ("b", u)]:
        # Clockwise vertices of a unit square aligned with the connection.
        vertices = [(i, center[0]+s*u[0]+t*v[0], center[1]+s*u[1]+t*v[1])
                    for i,(s,t) in enumerate([(-.5,-.5),(-.5,.5),(.5,.5),(.5,-.5)])]
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
        flopy.mf6.ModflowGwfdisv(gwf, nlay=1, ncpl=1, nvert=4, vertices=vertices,
                                cell2d=[(0,*center,4,0,1,2,3)], top=1., botm=0.)
        flopy.mf6.ModflowGwfic(gwf, strt=10.)
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k1, k22=k2, angle1=45.)
        if name == "a":
            flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[((0,0),10.)])
        else:
            flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[((0,0),-1.)])
        flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{name}.hds",
                               budget_filerecord=f"{name}.cbc",
                               saverecord=[("HEAD","ALL"),("BUDGET","ALL")])
    flopy.mf6.ModflowGwfgwf(sim, exgtype="GWF6-GWF6",
                           exgmnamea="b" if reverse else "a",
                           exgmnameb="a" if reverse else "b",
                           nexg=1, auxiliary=["ANGLDEGX"],
                           exchangedata=[((0,0),(0,0),1,.5,.5,1.,(angle+180*reverse)%360)])
    sim.write_simulation(silent=True)
    success, log = sim.run_simulation(silent=True, report=True)
    assert success, "\n".join(log)
    actual = float(flopy.utils.HeadFile(workspace / "b.hds").get_data().ravel()[0])
    # Default harmonic projection; both cells have identical conductivity,
    # unit thickness, unit interface width, unit center-to-center distance.
    phi=math.radians(45.)
    expected_k=1/(math.cos(theta-phi)**2/k1+math.sin(theta-phi)**2/k2)
    expected_head=10.-1./expected_k
    print(f"angle={angle} reverse={reverse} K=({k1},{k2}) head={actual:.12g} oracle={expected_head:.12g} delta={actual-expected_head:.12g}", flush=True)
    return abs(actual-expected_head)<1e-7

results=[]
with tempfile.TemporaryDirectory(prefix="mf01-") as tmp:
    for angle, k1, k2 in [(45,10.,1.),(135,10.,1.),(135,1.,1.),(0,10.,1.)]:
        for reverse in [False,True]:
            ws=pathlib.Path(tmp)/f"case-{len(results)}"
            passed=run_case(ws,angle,reverse,k1,k2)
            expected_pass = args.phase=="corrected" or not (angle==135 and k1!=k2)
            results.append((passed,expected_pass))
print(f"phase={args.phase}: {sum(p for p,e in results)}/{len(results)} oracle matches; {sum(p==e for p,e in results)}/{len(results)} expected outcomes")
assert all(p==e for p,e in results), results
