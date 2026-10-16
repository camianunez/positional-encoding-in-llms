"""Compute mathematical diagnostics and render the shared academic figure set."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if (ROOT/".vendor").exists():
    sys.path.insert(0,str(ROOT/".vendor"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scripts.figure_style import (configure,axes_style,save,BLUE,OCHRE,GREEN,
                                   LAVENDER,MUTED,RULE,BACKGROUND)

ASSETS=ROOT/"docs/assets"
RESULTS=ROOT/"experiments/diagnostics"

def freq(dim,base=10000):
    return base**(-np.arange(0,dim,2)/dim)

def kernel(delta,dim=64,base=10000):
    return np.cos(np.asarray(delta)[...,None]*freq(dim,base)).mean(-1)

def rotation(a):
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])

def block_summary(x,y,width=128):
    """Exact descriptive statistics per contiguous block; no sample uncertainty."""
    groups=[slice(i,min(i+width,len(x))) for i in range(0,len(x),width)]
    return tuple(np.asarray([fun(z[g]) for g in groups])
                 for z,fun in [(x,np.mean),(y,np.mean),(y,np.min),(y,np.max)])

def block_plot(ax,x,y,color,label):
    xb,mean,low,high=block_summary(x,y)
    ax.fill_between(xb,low,high,color=color,alpha=.14,linewidth=0)
    ax.plot(xb,mean,color=color,lw=1.7,label=label)
    return xb,mean

def main():
    configure()
    RESULTS.mkdir(parents=True,exist_ok=True)

    # Figure 1. Same vectors and transformation as the original diagnostic.
    fig,axs=plt.subplots(1,2,figsize=(9.8,4.0),layout="constrained")
    q,k=np.array([1.,0.]),np.array([0.85,0.53])
    for ax,shift,title in zip(axs,[0,1.4],["(a) Original positions","(b) Common position shift"]):
        ax.add_patch(plt.Circle((0,0),1,fill=False,color=RULE,lw=.7))
        ax.axhline(0,color=RULE,lw=.55)
        ax.axvline(0,color=RULE,lw=.55)
        for v,angle,label,color in [(q,.35,r"$R_s q$",BLUE),(k,1.05,r"$R_t k$",OCHRE)]:
            vec=rotation(angle+shift)@v
            ax.annotate("",xy=vec,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=color,lw=1.8,mutation_scale=13))
            ax.annotate(label,xy=vec,xytext=(9,8),textcoords="offset points",color=color,fontsize=13)
        score=(rotation(.35+shift)@q)@(rotation(1.05+shift)@k)
        ax.set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),aspect="equal",
               xlabel=r"$x_1$",ylabel=r"$x_2$",title=title)
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1])
        ax.text(.02,.02,f"Shift {shift:.1f} rad  ·  dot product {score:.3f}",
                transform=ax.transAxes,color=MUTED,fontsize=9)
        for spine in ax.spines.values():spine.set_visible(False)
    save(fig,"rotation",ASSETS)

    # Figure 2. A signed phase matrix replaces vertically offset wave traces.
    fig,axs=plt.subplots(1,2,figsize=(10.1,4.0),layout="constrained")
    p=np.arange(128)
    theta=freq(8)
    cmap=LinearSegmentedColormap.from_list("signed-phase",[BLUE,BACKGROUND,OCHRE])
    image=axs[0].pcolormesh(np.arange(129)-.5,np.arange(5)-.5,np.cos(theta[:,None]*p),
                           cmap=cmap,vmin=-1,vmax=1,shading="flat",antialiased=False)
    axs[0].set_xlim(-.5,127.5)
    axs[0].set_ylim(3.5,-.5)
    axs[0].set_yticks(range(4),[r"$\theta_0=1$",r"$\theta_1=0.1$",r"$\theta_2=0.01$",r"$\theta_3=0.001$"])
    axs[0].set_xticks([0,32,64,96,127])
    axs[0].set(title="(a) Rotary phase components",xlabel="Position p")
    cb=fig.colorbar(image,ax=axs[0],location="bottom",fraction=.045,pad=.12,aspect=35,ticks=[-1,0,1])
    cb.set_label(r"$\cos(p\theta_i)$",fontsize=10,labelpad=4)
    cb.solids.set_rasterized(False)
    ts=freq(64)
    axs[1].plot(np.arange(32),2*np.pi/ts,color=BLUE,marker="o",ms=3,lw=1.1)
    axs[1].set_yscale("log")
    axs[1].axhline(128,color=OCHRE,ls="--",lw=1)
    axs[1].text(30,165,"128 tokens",color=OCHRE,ha="right",fontsize=9)
    axs[1].set(title="(b) Frequency-dependent periods",xlabel="Coordinate-pair index i",
               ylabel=r"Period $2\pi/\theta_i$ (tokens)")
    axes_style(axs[1])
    save(fig,"frequencies",ASSETS)

    # Figure 3. Block means and exact within-block ranges reduce overplotting.
    delta=np.arange(8193)
    c=kernel(delta)
    partial=np.cumsum(np.exp(1j*delta[:,None]*freq(64)),axis=-1)
    abel=np.abs(partial).mean(-1)/32
    fig,axs=plt.subplots(1,2,figsize=(10.1,4.0),layout="constrained")
    block_plot(axs[0],delta,c,BLUE,"Block mean")
    axs[0].scatter([0],[1],s=16,color=OCHRE,zorder=4)
    axs[0].annotate(r"$C(0)=1$",xy=(0,1),xytext=(12,-4),textcoords="offset points",color=OCHRE,fontsize=10)
    axs[0].axhline(0,color=MUTED,lw=.65,ls="--")
    axs[0].set(title="(a) Positional similarity kernel",ylabel=r"$C(|\Delta|)$",ylim=(-.35,1.12))
    block_plot(axs[1],delta,abel,GREEN,"Block mean")
    axs[1].set(title="(b) Abel partial-sum magnitude",
               ylabel=r"$M^{-2}\sum_{j=1}^{M}|S_j|$",ylim=(0,.57))
    for ax in axs:
        axes_style(ax)
        ax.set(xlabel=r"Displacement $|\Delta|$ (tokens)",xlim=(0,8192))
        ax.set_xticks([0,2048,4096,6144,8192],["0","2,048","4,096","6,144","8,192"])
        ax.text(.97,.94,"128-offset blocks",transform=ax.transAxes,ha="right",va="top",color=MUTED,fontsize=9)
    save(fig,"decay",ASSETS)

    rng=np.random.default_rng(2026)
    theta=freq(64)
    q,k=rng.normal(size=(1024,32,2)),rng.normal(size=(1024,32,2))
    ds=np.linspace(0,10000,401)
    rms=[]
    for dist in ds:
        co,si=np.cos(dist*theta),np.sin(dist*theta)
        logits=(q[:,:,0]*(k[:,:,0]*co-k[:,:,1]*si)+q[:,:,1]*(k[:,:,0]*si+k[:,:,1]*co)).sum(-1)/8
        rms.append(np.sqrt(np.mean(logits**2)))
    fig,axs=plt.subplots(1,2,figsize=(10.1,4.0),layout="constrained")
    axs[0].plot(ds,rms,color=BLUE,lw=1.2)
    axs[0].axhline(1,color=OCHRE,ls="--",lw=1)
    axs[0].set(title="(a) Independent Gaussian content",xlabel=r"Displacement $|\Delta|$ (tokens)",
               ylabel="RMS scaled logit",ylim=(.85,1.15))
    axs[0].text(.03,.93,"Theoretical RMS = 1",transform=axs[0].transAxes,color=OCHRE,va="top",fontsize=9)
    near=np.arange(5872,6129)
    axs[1].plot(near,kernel(near-6000),color=BLUE,lw=1.4)
    axs[1].scatter([6000],[1],s=24,color=OCHRE,zorder=4)
    axs[1].set(title=r"(b) Constructed maximum at $\Delta=6000$",xlabel=r"Displacement $\Delta$ (tokens)",
               ylabel="Normalized score",ylim=(0,1.08))
    for ax in axs:axes_style(ax)
    axs[0].set_xticks([0,5000,10000],["0","5,000","10,000"])
    axs[1].set_xticks([5900,6000,6100],["5,900","6,000","6,100"])
    save(fig,"no-guaranteed-decay",ASSETS)

    fig,axs=plt.subplots(1,2,figsize=(10.1,4.0),layout="constrained")
    t=np.linspace(0,30,700)
    for u,color,label,style in [(0,BLUE,r"$u=0$", "--"),(-.06,GREEN,r"$u=-0.06$","-"),(.04,OCHRE,r"$u=0.04$","-")]:
        axs[0].plot(t,np.exp(u*t),color=color,ls=style,label=label)
    axs[0].set(title="(a) Generator amplitude",xlabel="Position parameter t",ylabel=r"Radial factor $e^{ut}$",ylim=(0,3.6))
    axs[0].legend(loc="upper left")
    origin=np.arange(31)
    axs[1].plot(origin,np.exp(-.06*(2*origin+3)),color=OCHRE)
    axs[1].axhline(1,color=BLUE,ls="--",lw=1.2)
    axs[1].text(29,1.025,"Orthogonal",color=BLUE,ha="right",fontsize=9)
    axs[1].text(14,.38,"Symmetric attenuation",color=OCHRE,fontsize=9)
    axs[1].set(title="(b) Fixed displacement, variable origin",xlabel="Query origin s",ylabel="Query–key amplitude factor",ylim=(0,1.14))
    for ax in axs:axes_style(ax)
    save(fig,"generators",ASSETS)

    bases=[10.,100.,10000.,1000000.]
    rows=[]
    ds=np.arange(128,16385)
    for base in bases:
        cc=kernel(ds,64,base)
        idx=int(np.argmax(cc))
        rows.append(dict(base=base,local_distance=float(np.sqrt(2-2*kernel(1,64,base))),
                         max_far_similarity=float(cc[idx]),far_offset=int(ds[idx]),domain=[128,16384],dim=64))
    fig,axs=plt.subplots(1,2,figsize=(10.1,4.0),layout="constrained")
    for ax,field,color,title,ylabel in [
        (axs[0],"local_distance",BLUE,"(a) Adjacent-position resolution",r"$\|\psi(1)-\psi(0)\|_2$"),
        (axs[1],"max_far_similarity",OCHRE,"(b) Maximum far-offset similarity",r"$\max_{128\leq\Delta\leq16384}C(\Delta)$")]:
        ax.plot(bases,[r[field] for r in rows],color=color,marker="o",ms=5,lw=1)
        ax.set_xscale("log")
        ax.set(title=title,xlabel="Frequency base B",ylabel=ylabel)
        axes_style(ax)
        ax.set_xticks(bases,[r"$10^1$",r"$10^2$",r"$10^4$",r"$10^6$"])
        ax.minorticks_off()
    save(fig,"base-tradeoff",ASSETS)

    fig,axs=plt.subplots(1,2,figsize=(10.1,4.0),layout="constrained")
    local=np.arange(129)
    axs[0].plot(local,np.cos(local*.1),color=BLUE,label="Original positions")
    axs[0].plot(local,np.cos(local*.1/4),color=OCHRE,ls="--",label="Positions / 4")
    axs[0].axvspan(0,32,alpha=.06,color=BLUE)
    axs[0].set(title="(a) Uniform position interpolation",xlabel="Position p",ylabel=r"$\cos(p\theta/a)$")
    axs[0].legend(loc="upper center",bbox_to_anchor=(.5,-.25),ncol=2)
    delta=np.arange(129)
    logits=4*kernel(delta)
    for n,color in [(16,BLUE),(64,OCHRE),(128,GREEN)]:
        prob=np.exp(logits[:n]-logits[:n].max());prob/=prob.sum()
        axs[1].plot(delta[:n],prob,color=color,label=f"{n} keys")
    axs[1].set(title="(b) Softmax over variable key counts",xlabel="Causal lag r (tokens)",ylabel="Attention probability")
    axs[1].legend(loc="upper center",bbox_to_anchor=(.5,-.25),ncol=3)
    for ax in axs:axes_style(ax)
    save(fig,"scaling-softmax",ASSETS)

    diagnostics=dict(spectrum=rows,kernel=dict(dim=64,base=10000,zero=float(kernel(0)),
        at_128=float(kernel(128)),at_8192=float(kernel(8192))),
        random_content=dict(draws=1024,seed=2026,min_rms=float(min(rms)),max_rms=float(max(rms))),
        interpretation="Geometry diagnostics are not trained-model quality metrics.")
    (RESULTS/"geometry.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    print("Rendered seven mathematical figures as SVG and PNG with the shared figure specification.")

if __name__=="__main__":
    main()
