"""Validate raw runs and regenerate all published experimental tables and plots."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if (ROOT/".vendor").exists():
    sys.path.insert(0,str(ROOT/".vendor"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rope_lab.model import METHODS
from scripts.train import source_hash
from scripts.figure_style import configure,axes_style,save,METHOD_COLORS,METHOD_MARKERS,MUTED,BACKGROUND

LABELS={"none":"No explicit PE","learned":"Learned absolute","sinusoidal":"Additive sinusoidal",
        "rope":"RoPE","single_rope":"Single-frequency RoPE","alibi":"ALiBi","rope4":"RoPE / 4 (frozen)"}
ORDER=list(METHODS)+["rope4"]
OUT=ROOT/"experiments/results"
ASSETS=ROOT/"docs/assets"

def collect():
    env=json.loads((OUT/"environment.json").read_text())
    assert env["source_sha256"]==source_hash(),"training sources differ from the recorded experiment"
    assert env["protocol"]["steps"]==1200 and env["protocol"]["seeds"]==[11,22,33]
    records=[]
    for method in METHODS:
        for seed in [11,22,33]:
            run=json.loads((OUT/f"{method}-{seed}.json").read_text())
            assert run["method"]==method and run["seed"]==seed
            assert run["source_sha256"]==env["source_sha256"]
            assert run["config"]==dict(method=method,vocab=84,dim=64,heads=4,layers=2,max_positions=256,base=10000.0)
            expected=12 if method=="rope" else 6
            assert len(run["metrics"])==expected
            seen=set()
            for m in run["metrics"]:
                identity=(m["task"],m["length"],m["scale"])
                assert identity not in seen
                seen.add(identity)
                assert m["length"] in ([32,64,128] if m["task"]=="lag4" else [31,63,127])
                assert m["examples"]==768 and 0<=m["correct"]<=768
                assert abs(m["accuracy"]-m["correct"]/768)<1e-12
                assert np.isfinite(m["nll"]) and m["nll"]>=0
                assert abs(np.exp(m["nll"])-m["answer_perplexity"])<1e-8
                assert m["scale"] in ([1.,4.] if method=="rope" else [1.])
                records.append(dict(method="rope4" if m["scale"]==4 else method,seed=seed,**m))
    assert len(records)==126
    summary=[]
    for task,lengths in [("lag4",[32,64,128]),("lookup",[31,63,127])]:
        for method in ORDER:
            for length in lengths:
                rows=[r for r in records if r["task"]==task and r["method"]==method and r["length"]==length]
                assert len(rows)==3
                item=dict(task=task,method=method,length=length,seeds=[r["seed"] for r in rows])
                for metric in ["accuracy","nll","answer_perplexity"]:
                    values=[r[metric] for r in rows]
                    item[metric+"_mean"]=statistics.mean(values)
                    item[metric+"_sd"]=statistics.stdev(values)
                summary.append(item)
    return records,summary

def chapter(summary):
    def cell(task,method,length,metric,factor=1):
        r=next(x for x in summary if x["task"]==task and x["method"]==method and x["length"]==length)
        return f'{factor*r[metric+"_mean"]:.2f} ± {factor*r[metric+"_sd"]:.2f}'
    text="""# 9. Empirical results and limitations

## Evaluation protocol and summary statistics

The recorded experiment comprises 18 decoders: six positional methods and three training seeds. Each task/length condition contains 768 held-out answers per seed; the held-out examples are shared across methods and training seeds. Tables report **mean ± sample standard deviation across seeds**. These are descriptive summaries rather than confidence intervals. The [experimental protocol](08-evaluation.md) specifies all controls and limitations.

![Answer accuracy by positional method, task, and input length](assets/benchmark-accuracy.svg)

*Figure 8. Exact answer accuracy for each method at three input lengths. Panels separate tasks and context lengths to avoid overlapping trajectories. Points show means across three training seeds; horizontal bars show ± one sample standard deviation. The vertical dashed line is uniform guessing over 16 valid answer values (6.25%). The first column uses a training input length.*

"""
    for task,lengths,title in [("lag4",[32,64,128],"Fixed-lag copying: accuracy and conditional perplexity"),
                               ("lookup",[31,63,127],"Associative lookup: accuracy and conditional perplexity")]:
        text+=f"## {title}\n\n"
        text+="| Method | "+" | ".join(f"{n} tokens: accuracy %" for n in lengths)+" |\n"
        text+="|---|---:|---:|---:|\n"
        for method in ORDER:
            text+="| "+LABELS[method]+" | "+" | ".join(cell(task,method,n,"accuracy",100) for n in lengths)+" |\n"
        text+="\n| Method | "+" | ".join(f"{n} tokens: answer PPL" for n in lengths)+" |\n"
        text+="|---|---:|---:|---:|\n"
        for method in ORDER:
            text+="| "+LABELS[method]+" | "+" | ".join(cell(task,method,n,"answer_perplexity") for n in lengths)+" |\n"
        text+="\n"
    text+="""## Interpretation of method and context-length effects

**Training-length copying.** Learned positions, additive sinusoids, and multiscale RoPE attain 100% held-out copying accuracy at 32 tokens in these runs. The decoder without explicit positional encoding performs substantially worse under the same training procedure.

**Length extrapolation.** RoPE satisfies the norm, common-shift, gradient, and cache checks but loses copying accuracy at longer lengths. The mathematical invariants therefore do not determine the length-generalization behavior learned under this protocol.

**Frequency ablation.** Multiscale RoPE outperforms the single-frequency variant at the training length, while the single-frequency variant retains higher copying accuracy at the longer tested lengths. The ranking depends on the evaluation domain and finite training procedure.

**Frozen position interpolation.** Dividing positions by four reduces copying accuracy in the frozen RoPE checkpoints. The intervention changes the phase representation of the required four-token lag. No adaptation is performed, so this condition is distinct from published fine-tuned interpolation procedures.

**Associative lookup.** Every method remains substantially below oracle performance, including at the training length. Above-uniform accuracy can reflect value frequencies or other shortcuts and does not establish reliable key–value binding.

## Conditional answer perplexity

![Conditional answer perplexity by method and context length](assets/benchmark-ppl.svg)

*Figure 9. Conditional answer perplexity on the held-out answer token, using full-vocabulary logits. The horizontal axis is logarithmic. Points show means across seeds; bars show ± one sample standard deviation, clipped at the natural lower bound of 1 where necessary. The vertical dashed reference is perplexity 16, corresponding to a predictor uniform over valid answer values. These are not corpus language-model perplexities.*

## Optimization trajectories on associative lookup

![Lookup minibatch loss summarized across training seeds](assets/training-curves.svg)

*Figure 10. Lookup-minibatch negative log likelihood recorded every 100 total training updates. Lines show the mean across three seeds; shaded regions show their minimum and maximum. Each observation uses a newly sampled batch. These ranges describe the recorded seeds and are not confidence intervals. The dashed reference is log(16).*

## Scope of empirical conclusions

The results demonstrate that explicit position information can support precise local addressing under this training budget. They also demonstrate that the best observed method changes with the evaluation length. None of the tested methods provides uniformly reliable behavior across all conditions.

The unresolved lookup task, absence of method-specific tuning, small model size, and synthetic data restrict the conclusions. Selection for a production LLM requires natural-text evaluation and larger-scale controlled comparisons.

The geometric base sweep in [section 7](07-long-context.md) measures phase properties. It does not determine the best base for these trained models, whose conventional RoPE base is fixed.

## Recorded measurements and provenance

- [Per-seed CSV measurements](https://github.com/camianunez/positional-encoding-in-llms/blob/main/experiments/results/metrics.csv)
- [Aggregated JSON statistics](https://github.com/camianunez/positional-encoding-in-llms/blob/main/experiments/results/summary.json)
- [Environment and protocol](https://github.com/camianunez/positional-encoding-in-llms/blob/main/experiments/results/environment.json)
- [Individual training-run records](https://github.com/camianunez/positional-encoding-in-llms/tree/main/experiments/results)

The report verifies method and seed identities, source hashes, metric counts, held-out example counts, and the NLL/perplexity relation before generating the tables. All figures use the shared rendering specification and the recorded measurements.

[Section 10: Reference implementation and numerical verification →](10-implementation.md)
"""
    return text

def csv_text(records):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=["method","seed","task","length","scale","examples","correct","accuracy","nll","answer_perplexity"],lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()

def figures(records,summary):
    configure()
    ASSETS.mkdir(parents=True,exist_ok=True)
    short={"none":"No explicit PE","learned":"Learned absolute","sinusoidal":"Sinusoidal",
           "rope":"RoPE","single_rope":"Single frequency","alibi":"ALiBi","rope4":"RoPE / 4"}
    for metric,name in [("accuracy","benchmark-accuracy"),("answer_perplexity","benchmark-ppl")]:
        fig,axs=plt.subplots(2,3,figsize=(10.5,6.7),sharex=True,sharey=True,layout="constrained")
        if metric=="answer_perplexity":
            upper=max(r[metric+"_mean"]+r[metric+"_sd"] for r in summary)*1.25
        for row,(task,lengths,title) in enumerate([("lag4",[32,64,128],"Lag-four copying"),
                                                   ("lookup",[31,63,127],"Associative lookup")]):
            for col,length in enumerate(lengths):
                ax=axs[row,col]
                if metric=="answer_perplexity":
                    ax.set_xscale("log")
                for index,method in enumerate(ORDER):
                    result=next(r for r in summary if r["task"]==task and r["method"]==method and r["length"]==length)
                    factor=100 if metric=="accuracy" else 1
                    mean=result[metric+"_mean"]*factor
                    sd=result[metric+"_sd"]*factor
                    low=max(0 if metric=="accuracy" else 1,mean-sd)
                    high=min(100,mean+sd) if metric=="accuracy" else mean+sd
                    ax.errorbar(mean,index,xerr=np.array([[mean-low],[high-mean]]),
                        fmt=METHOD_MARKERS[method],color=METHOD_COLORS[method],ms=4.5,
                        elinewidth=1,capsize=2,capthick=.8,zorder=4)
                ax.axvline(6.25 if metric=="accuracy" else 16,color=MUTED,ls="--",lw=.75,alpha=.8)
                axes_style(ax,grid="x",ticks=4)
                ax.set_yticks(np.arange(len(ORDER)),[short[m] for m in ORDER])
                ax.set_ylim(len(ORDER)-.45,-.55)
                ax.set_title(f"{title}\n{length} input tokens",fontsize=10.4,loc="left",pad=12)
                ax.tick_params(axis="y",length=0,labelsize=9.4,labelleft=col==0)
                ax.spines["left"].set_visible(False)
                if metric=="accuracy":
                    ax.set_xlim(0,104)
                    ax.set_xticks([0,25,50,75,100])
                else:
                    ax.set_xlim(1,upper)
                if row==1:
                    ax.set_xlabel("Accuracy (%)" if metric=="accuracy" else "Conditional answer perplexity",fontsize=9.5)
        save(fig,name,ASSETS)

    fig,axs=plt.subplots(2,3,figsize=(10.5,5.8),sharex=True,sharey=True,layout="constrained")
    for ax,method in zip(axs.flat,METHODS):
        runs=[json.loads((OUT/f"{method}-{seed}.json").read_text()) for seed in [11,22,33]]
        steps=np.asarray([p["step"] for p in runs[0]["curve"]])
        values=np.asarray([[p["loss"] for p in run["curve"]] for run in runs])
        color=METHOD_COLORS[method]
        ax.fill_between(steps,values.min(axis=0),values.max(axis=0),color=color,alpha=.14,linewidth=0)
        ax.plot(steps,values.mean(axis=0),color=color,lw=1.5)
        ax.axhline(np.log(16),ls="--",color=MUTED,lw=.75)
        ax.set_title(LABELS[method],fontsize=10.5,loc="left")
        axes_style(ax,ticks=3)
        ax.set_xticks([0,400,800,1200],["0","400","800","1,200"])
        ax.set_xlabel("Total updates",fontsize=9.5)
    for ax in axs[:,0]:
        ax.set_ylabel("Lookup batch NLL (nats)",fontsize=10)
    save(fig,"training-curves",ASSETS)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    records,summary=collect()
    generated={ROOT/"docs/09-results.md":chapter(summary),
               OUT/"summary.json":json.dumps(summary,indent=2)+"\n",
               OUT/"metrics.csv":csv_text(records)}
    if args.check:
        for path,content in generated.items():
            assert path.read_text(encoding="utf-8")==content,f"stale generated file: {path}"
        print("All 18 runs, 126 metric records, 42 summaries and generated tables verified.")
    else:
        for path,content in generated.items():
            path.write_text(content,encoding="utf-8",newline="\n")
        figures(records,summary)
        provenance=dict(numpy=np.__version__,matplotlib=matplotlib.__version__,
            report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            diagnostics_source_sha256=hashlib.sha256((ROOT/"scripts/diagnostics.py").read_bytes()).hexdigest(),
            style_source_sha256=hashlib.sha256((ROOT/"scripts/figure_style.py").read_bytes()).hexdigest(),
            preferred_format="SVG",background=BACKGROUND)
        (ROOT/"experiments/diagnostics/figure-environment.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
        print("Generated academic result tables and three figures from the recorded runs.")

if __name__=="__main__":
    main()
