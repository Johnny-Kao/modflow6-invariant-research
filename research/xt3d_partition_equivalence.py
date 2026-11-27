"""System-level XT3D equivalence: unchanged physics, different decompositions."""
import argparse
from pathlib import Path
import tempfile
import re
import numpy as np
import flopy

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=Path)
args = parser.parse_args()
exe = str(args.exe.resolve())
NL, NR, NC = 2, 6, 6
D = 10.0

def run(ws, pr, pc, omit, tilted, reverse=False):
    sim = flopy.mf6.MFSimulation(sim_name="partition", sim_ws=ws, exe_name=exe)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1.,1,1.)])
    ims = flopy.mf6.ModflowIms(sim, print_option="SUMMARY", outer_dvclose=1e-10,
                              inner_dvclose=1e-12, rcloserecord=1e-10,
                              outer_maximum=200, inner_maximum=500,
                              linear_acceleration="BICGSTAB")
    nr, nc = NR//pr, NC//pc
    names = []
    for r in range(pr):
        for c in range(pc):
            name = f"m{r}{c}"
            names.append(name)
            shape = (NL,nr,nc)
            col = np.broadcast_to(np.arange(c*nc,(c+1)*nc),shape)
            row = np.broadcast_to(np.arange(r*nr,(r+1)*nr)[None,:,None],shape)
            angle1 = np.where(col>=NC//2,30.,0.)
            angle2 = np.where((col>=NC//2)&tilted,45.,0.)
            angle3 = np.where((col>=NC//2)&tilted,15.,0.)
            k = np.where(col>=NC//2,10.,2.) * (1.+0.1*row)
            gwf = flopy.mf6.ModflowGwf(sim,modelname=name,save_flows=True)
            flopy.mf6.ModflowGwfdis(gwf,nlay=NL,nrow=nr,ncol=nc,
                delr=D,delc=D,top=100.,botm=[90.,80.],
                xorigin=c*nc*D,yorigin=(NR-(r+1)*nr)*D)
            flopy.mf6.ModflowGwfic(gwf,strt=105.)
            flopy.mf6.ModflowGwfnpf(gwf,icelltype=0,k=k,k22=k/10.,k33=k/5.,
                angle1=angle1,angle2=None if omit and not angle2.any() else angle2,
                angle3=None if omit and not angle2.any() else angle3,
                xt3doptions=True,save_flows=True,save_specific_discharge=True)
            chd = []
            if c==0:
                chd.extend(((l,i,0),110.-2*l) for l in range(NL) for i in range(nr))
            if c==pc-1:
                chd.extend(((l,i,nc-1),100.+2*l) for l in range(NL) for i in range(nr))
            if chd:
                flopy.mf6.ModflowGwfchd(gwf,stress_period_data=chd)
            wr,wc = 2,2
            if r*nr<=wr<(r+1)*nr and c*nc<=wc<(c+1)*nc:
                flopy.mf6.ModflowGwfwel(gwf,stress_period_data=[((0,wr-r*nr,wc-c*nc),-1.)])
            flopy.mf6.ModflowGwfoc(gwf,head_filerecord=f"{name}.hds",
                budget_filerecord=f"{name}.cbc",
                saverecord=[("HEAD","ALL"),("BUDGET","ALL")])
    sim.register_ims_package(ims,names)
    def exchange(a,b,data):
        if reverse:
            a,b=b,a
            data=[(v[1],v[0],v[2],v[4],v[3],v[5],(v[6]+180.)%360.,v[7]) for v in data]
        flopy.mf6.ModflowGwfgwf(sim,exgtype="GWF6-GWF6",exgmnamea=a,
            exgmnameb=b,nexg=len(data),exchangedata=data,xt3d=True,
            auxiliary=["ANGLDEGX","CDIST"],filename=f"{a}-{b}.exg")
    for r in range(pr):
        for c in range(pc):
            a=f"m{r}{c}"
            if c+1<pc:
                exchange(a,f"m{r}{c+1}",[((l,i,nc-1),(l,i,0),1,D/2,D/2,D,0.,D)
                                        for l in range(NL) for i in range(nr)])
            if r+1<pr:
                exchange(a,f"m{r+1}{c}",[((l,nr-1,j),(l,0,j),1,D/2,D/2,D,270.,D)
                                        for l in range(NL) for j in range(nc)])
    sim.write_simulation(silent=True)
    ok, log=sim.run_simulation(silent=True,report=True)
    if not ok:
        print("\n".join(log),flush=True)
        print((ws/"mfsim.lst").read_text()[-5000:],flush=True)
        raise RuntimeError("model did not finish")
    heads=np.empty((NL,NR,NC))
    discharge=np.empty((3,NL,NR,NC))
    maxres=0.
    for r in range(pr):
        for c in range(pc):
            name=f"m{r}{c}"
            loc=(slice(None),slice(r*nr,(r+1)*nr),slice(c*nc,(c+1)*nc))
            heads[loc]=flopy.utils.HeadFile(ws/f"{name}.hds").get_data()
            b=flopy.utils.CellBudgetFile(ws/f"{name}.cbc",precision="double")
            data=b.get_data(text="DATA-SPDIS")[-1]
            for i,field in enumerate(("qx","qy","qz")):
                vals=np.zeros(NL*nr*nc)
                vals[data["node"]-1]=data[field]
                discharge[(i,*loc)]=vals.reshape(NL,nr,nc)
            lines=(ws/f"{name}.lst").read_text()
            for value in re.findall(r"PERCENT DISCREPANCY\s*=\s*([-0-9.E+]+)",lines):
                maxres=max(maxres,abs(float(value)))
    assert np.isfinite(heads).all() and np.isfinite(discharge).all()
    return heads,discharge,maxres

with tempfile.TemporaryDirectory(prefix="xt3d-partition-") as tmp:
    root=Path(tmp)
    for tilted in (False,True):
        base=run(root/f"mono-{tilted}",1,1,False,tilted)
        print(f"monolithic tilted={tilted} head_range={np.ptp(base[0]):.12g} budget_percent={base[2]:.12g}",flush=True)
        for pr,pc in ((1,2),(2,2),(3,3)):
            for omit,reverse in ((False,False),(True,False),(True,True)):
                case=f"split-{tilted}-{pr}-{pc}-{omit}-{reverse}"
                try:
                    h,q,b=run(root/case,pr,pc,omit,tilted,reverse)
                    print(f"case={case} head_max_abs={np.max(abs(h-base[0])):.12g} discharge_max_abs={np.max(abs(q-base[1])):.12g} budget_percent={b:.12g}",flush=True)
                except Exception as exc:
                    print(f"case={case} ERROR {type(exc).__name__}: {exc}",flush=True)
                    raise
