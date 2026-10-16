"""Shared academic figure specification for the permanent dark edition."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BACKGROUND="#171c23"
PANEL=BACKGROUND
TEXT="#dce1e7"
MUTED="#a5afbb"
RULE="#414c59"
BLUE="#9bbdd8"
OCHRE="#d8b984"
GREEN="#9dbfae"
LAVENDER="#b9add3"
ROSE="#cf9fa5"
SLATE="#9aa6b6"
COLORS=[BLUE,OCHRE,GREEN,LAVENDER,ROSE,SLATE,"#b2ced0"]
METHOD_COLORS={"none":SLATE,"learned":ROSE,"sinusoidal":OCHRE,"rope":BLUE,
               "single_rope":LAVENDER,"alibi":GREEN,"rope4":"#b2ced0"}
METHOD_MARKERS={"none":"o","learned":"s","sinusoidal":"D","rope":"o",
                "single_rope":"^","alibi":"v","rope4":"P"}

def configure():
    plt.rcParams.update({
        "figure.dpi":140,"savefig.dpi":240,"figure.facecolor":BACKGROUND,
        "axes.facecolor":PANEL,"savefig.facecolor":BACKGROUND,
        "text.color":TEXT,"axes.labelcolor":TEXT,"xtick.color":MUTED,"ytick.color":MUTED,
        "axes.edgecolor":RULE,"axes.linewidth":.65,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":False,"grid.color":RULE,"grid.alpha":.38,"grid.linewidth":.5,
        "font.family":"DejaVu Sans","font.size":11,"axes.labelsize":10.5,
        "xtick.labelsize":9.5,"ytick.labelsize":9.5,"axes.titlesize":11.5,
        "axes.titleweight":"normal","axes.titlelocation":"left","axes.titlepad":13,
        "legend.fontsize":9.2,"legend.frameon":False,"legend.labelcolor":TEXT,
        "lines.linewidth":1.5,"lines.markersize":4.5,"mathtext.fontset":"stix",
        "svg.fonttype":"path","svg.hashsalt":"positional-encoding-in-llms",
        "axes.prop_cycle":plt.cycler(color=COLORS),
        "axes.formatter.useoffset":False,
    })

def axes_style(ax, *, grid="y", ticks=4):
    ax.grid(axis=grid)
    if ax.get_xscale()=="linear":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=ticks))
    if ax.get_yscale()=="linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ticks))
    ax.tick_params(length=3,width=.6,pad=5)
    ax.margins(x=.025)
    return ax

def save(fig,name,directory):
    directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/f"{name}.svg",bbox_inches="tight",metadata={"Date":None})
    fig.savefig(directory/f"{name}.png",bbox_inches="tight")
    plt.close(fig)
