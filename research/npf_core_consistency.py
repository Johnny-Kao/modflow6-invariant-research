"""Baseline checks of final saturation and equivalent conductivity inputs."""
import argparse
from pathlib import Path
import tempfile
import flopy
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("exe", type=Path)
args = p.parse_args()
exe = str(args.exe.resolve())


def make_sim(ws, ncol, top, start, tol, newton=False):
    sim = flopy.mf6.MFSimulation(sim_name="npf", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1., 1, 1.)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True,
                              newtonoptions="NEWTON" if newton else None)
    ims = flopy.mf6.ModflowIms(sim, outer_dvclose=tol, outer_maximum=100,
                               inner_dvclose=1e-12, inner_maximum=100,
                               rcloserecord=1e-10, linear_acceleration="BICGSTAB")
    sim.register_ims_package(ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=ncol,
                           delr=1., delc=1., top=top, botm=0.)
    flopy.mf6.ModflowGwfic(gwf, strt=start)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds",
                           budget_filerecord="gwf.cbc",
                           saverecord=[("HEAD","ALL"),("BUDGET","ALL")])
    return sim, gwf


def solve(sim, ws):
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    assert ok, "\n".join(log)
    return flopy.utils.HeadFile(ws / "gwf.hds").get_data().ravel()


def saturation(ws, top, tol, newton):
    start = top * 0.5
    target = start + 0.005
    sim, gwf = make_sim(ws, 1, top, start, tol, newton)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=1., save_saturation=True)
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=[((0,0,0), target, 1.)])
    head = float(solve(sim, ws)[0])
    assert abs(head-target) < 1e-7, (head,target)
    budget = flopy.utils.CellBudgetFile(ws / "gwf.cbc", precision="double")
    sat = float(budget.get_data(text="DATA-SAT")[-1]["sat"][0])
    br = head / top
    expected = (br-0.5e-6)/(1.-1e-6) if newton else br
    print(f"SAT top={top} tol={tol} newton={newton} head={head:.12g} sat={sat:.12g} expected={expected:.12g} delta={sat-expected:.12g}", flush=True)


def anisotropy(ws, angle, explicit):
    sim, gwf = make_sim(ws, 2, 1., 10., 1e-10)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10., k33=1.,
                           k22=10. if explicit else None,
                           angle1=0., angle2=angle, angle3=0.)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[((0,0,0),10.)])
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[((0,0,1),-1.)])
    head = float(solve(sim, ws)[1])
    expected = 10. - (np.cos(np.deg2rad(angle))**2/10. + np.sin(np.deg2rad(angle))**2)
    print(f"ANISO angle={angle} explicit_k22={explicit} head={head:.12g} expected={expected:.12g} delta={head-expected:.12g}", flush=True)


with tempfile.TemporaryDirectory(prefix="npf-core-") as tmp:
    root = Path(tmp)
    for top in (1., 0.1):
        for tol in (0.01, 1e-10):
            for newton in (False, True):
                saturation(root / f"sat-{top}-{tol}-{newton}", top, tol, newton)
    for angle in (0., 45., 90.):
        for explicit in (False, True):
            anisotropy(root / f"aniso-{angle}-{explicit}", angle, explicit)
