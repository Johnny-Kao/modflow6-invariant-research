"""Compare equivalent absolute and relative delay-bed initial heads."""
import argparse
import pathlib
import tempfile
import numpy as np
import flopy

p = argparse.ArgumentParser()
p.add_argument("exe", type=pathlib.Path)
p.add_argument("phase", choices=["baseline", "corrected"])
args = p.parse_args()
exe = str(args.exe.resolve())


def run(ws, offset, absolute, newton):
    sim = flopy.mf6.MFSimulation(sim_name="csub", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1,
                         perioddata=[(0.01, 1, 1.0)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True,
                              newtonoptions="NEWTON" if newton else None)
    ims = flopy.mf6.ModflowIms(sim, outer_dvclose=1e-10, inner_dvclose=1e-12,
                              rcloserecord=1e-10, outer_maximum=200,
                              inner_maximum=200, linear_acceleration="BICGSTAB")
    sim.register_ims_package(ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=1,
                           delr=1., delc=1., top=0., botm=-100.)
    flopy.mf6.ModflowGwfic(gwf, strt=10.)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0., sy=0., transient={0:True})
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=[((0,0,0),10.,100.)])
    csub = flopy.mf6.ModflowGwfcsub(
        gwf, head_based=True, specified_initial_delay_head=absolute,
        ndelaycells=7, ninterbeds=1, cg_theta=0.3, cg_ske_cr=0.,
        beta=0., sgm=1.7, sgs=2.,
        packagedata=[(0,(0,0,0),"delay",0.,20.,1.,0.001,0.001,0.3,0.0001,
                      10.+offset if absolute else offset)],
        save_flows=True,
    )
    csub.obs.initialize(digits=16, continuous={"csub.csv":[
        ("middle","delay-head",(0,3)),
        ("comp","interbed-compaction",(0,)),
        ("topflow","delay-flowtop",(0,)),
        ("botflow","delay-flowbot",(0,)),
    ]})
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds",
                           budget_filerecord="gwf.cbc",
                           saverecord=[("HEAD","ALL"),("BUDGET","ALL")])
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    if not ok:
        print("\n".join(log))
        print((ws / "mfsim.lst").read_text()[-8000:])
        raise AssertionError("native simulation did not complete")
    out = np.genfromtxt(ws / "csub.csv", delimiter=",", names=True)
    h = float(np.asarray(flopy.utils.HeadFile(ws / "gwf.hds").get_data()).ravel()[0])
    names = {n.lower(): n for n in out.dtype.names}
    vals = [float(out[names[n]]) for n in ("middle","comp","topflow","botflow")]
    print(f"offset={offset} absolute={absolute} newton={newton} gwf_head={h:.12g} delay_center={vals[0]:.12g} compaction={vals[1]:.12g} topflow={vals[2]:.12g} botflow={vals[3]:.12g}", flush=True)
    # A short step relative to the diffusion timescale cannot erase an
    # imposed initial pressure offset at the middle of a thick delay bed.
    return np.array([h, *vals])


with tempfile.TemporaryDirectory(prefix="csub-head-") as tmp:
    root = pathlib.Path(tmp)
    results = []
    for newton in (False, True):
        for offset in (0., -3., 3.):
            rel = run(root / f"relative-{newton}-{offset}", offset, False, newton)
            absolute = run(root / f"absolute-{newton}-{offset}", offset, True, newton)
            assert abs(rel[1] - (10.+offset)) < 1e-3, (offset, rel)
            match = bool(np.allclose(rel, absolute, rtol=1e-7, atol=1e-8))
            expected = args.phase == "corrected" or offset == 0.
            results.append((match, expected))
            print(f"pair offset={offset} newton={newton} match={match} max_delta={np.max(np.abs(rel-absolute)):.12g}", flush=True)
    print(f"{args.phase}: {sum(m for m,e in results)}/{len(results)} equivalent input pairs match", flush=True)
    assert all(m==e for m,e in results), results
