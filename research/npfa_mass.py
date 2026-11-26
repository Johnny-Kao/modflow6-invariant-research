"""Independent water and solute inventory oracle for an ordinary unconfined cell."""
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


def run(ws, start, delta, tol):
    porosity = 0.25
    q = porosity * delta
    c0 = 100.0
    sim = flopy.mf6.MFSimulation(sim_name="sat", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1., 1, 1.)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    gwt = flopy.mf6.ModflowGwt(sim, modelname="gwt", save_flows=True)
    for model, dvclose in ((gwf, tol), (gwt, 1e-10)):
        ims = flopy.mf6.ModflowIms(
            sim, filename=model.name+".ims", outer_dvclose=dvclose,
            inner_dvclose=1e-13, rcloserecord=1e-12,
            outer_maximum=100, inner_maximum=100, linear_acceleration="BICGSTAB")
        sim.register_ims_package(ims, [model.name])
    geom = dict(nlay=1, nrow=1, ncol=1, delr=1., delc=1., top=1., botm=0.)
    flopy.mf6.ModflowGwfdis(gwf, **geom)
    flopy.mf6.ModflowGwfic(gwf, strt=start)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=1., save_saturation=True)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=0., sy=porosity, transient={0:True})
    flopy.mf6.ModflowGwfwel(gwf, pname="WEL-1", auxiliary=["CONCENTRATION"],
                           stress_period_data=[((0,0,0), q, 0.)], save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds",
                           budget_filerecord="gwf.cbc",
                           saverecord=[("HEAD","ALL"),("BUDGET","ALL")])
    flopy.mf6.ModflowGwtdis(gwt, **geom)
    flopy.mf6.ModflowGwtic(gwt, strt=c0)
    flopy.mf6.ModflowGwtadv(gwt)
    flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
    flopy.mf6.ModflowGwtssm(gwt, sources=[("WEL-1","AUX","CONCENTRATION")])
    flopy.mf6.ModflowGwtoc(gwt, concentration_filerecord="gwt.ucn",
                           budget_filerecord="gwt.cbc",
                           saverecord=[("CONCENTRATION","ALL"),("BUDGET","ALL")])
    flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6",
                           exgmnamea="gwf", exgmnameb="gwt")
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    assert ok, "\n".join(log)
    head = float(flopy.utils.HeadFile(ws/"gwf.hds").get_data().ravel()[0])
    conc = float(flopy.utils.HeadFile(ws/"gwt.ucn", text="CONCENTRATION").get_data().ravel()[0])
    b = flopy.utils.CellBudgetFile(ws/"gwf.cbc", precision="double")
    sat = float(b.get_data(text="DATA-SAT")[-1]["sat"][0])
    sy = float(np.asarray(b.get_data(text="STO-SY", full3D=True)[-1]).sum())
    well = float(b.get_data(text="WEL")[-1]["q"].sum())
    assert abs(head - (start+delta)) < 1e-9, (head,start,delta)
    assert abs(sy + well) < 1e-10, (sy,well)
    expected = c0 * start / head
    mass_ratio = conc * head / (c0 * start)
    print(f"NPFA start={start} delta={delta} tol={tol} head={head:.12g} sat={sat:.12g} "
          f"conc={conc:.12g} expected={expected:.12g} mass_ratio={mass_ratio:.12g} water_residual={sy+well:.12g}", flush=True)
    consistent = abs(sat-head) < 1e-10 and abs(mass_ratio-1.) < 1e-9
    return consistent


with tempfile.TemporaryDirectory(prefix="npfa-") as tmp:
    root = Path(tmp)
    outcomes = []
    for start in (0.5, 0.05, 0.01, 0.005):
        for tol in (0.01, 1e-10):
            outcome = run(root/f"{start}-{tol}", start, 0.005, tol)
            outcomes.append(outcome)
            assert outcome == (args.phase=="corrected" or tol < 1e-8), (start,tol,outcome)
    print(f"{args.phase}: {sum(outcomes)}/{len(outcomes)} physical inventory and saturation matches", flush=True)
