"""Controlled decoder comparison with per-run resumption and source hashes."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from rope_lab.model import METHODS,ModelConfig,TinyDecoder
from rope_lab.tasks import batch

def source_hash():
    files = sorted((ROOT/"rope_lab").glob("*.py"))+[Path(__file__)]
    return hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()

@torch.no_grad()
def evaluate(model,task,length,batches,size,device,scale=1.0):
    model.eval()
    # Identical held-out stream for every method and training seed.
    g = torch.Generator().manual_seed(900000+length+(1000 if task=="lookup" else 0))
    loss,correct,count = 0.0,0,0
    for _ in range(batches):
        x,y = batch(task,size,length,g,device)
        logits = model(x,scale=scale)[0][:,-1]
        loss += F.cross_entropy(logits,y,reduction="sum").item()
        correct += (logits.argmax(-1)==y).sum().item()
        count += size
    model.train()
    nll = loss/count
    return dict(task=task,length=length,examples=count,correct=correct,accuracy=correct/count,
                nll=nll,answer_perplexity=math.exp(nll),scale=scale)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="experiments/results")
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--seeds",type=int,nargs="+",default=[11,22,33])
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    p.add_argument("--batch-size",type=int,default=48)
    p.add_argument("--eval-batches",type=int,default=16)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    provenance = dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),
        processor=platform.processor(),device=args.device,threads=args.threads,
        source_sha256=source_hash(),command=sys.argv,
        protocol=dict(steps=args.steps,seeds=args.seeds,batch_size=args.batch_size,
            eval_batches=args.eval_batches,lr=0.001,weight_decay=0.01,
            task_schedule="alternating lag4 and lookup",
            train_lag4_lengths=[16,24,32],train_lookup_lengths=[15,23,31]))
    env_path = out/"environment.json"
    if env_path.exists():
        old = json.loads(env_path.read_text())
        if old["source_sha256"]!=provenance["source_sha256"] or old["protocol"]!=provenance["protocol"]:
            raise RuntimeError("output has different protocol/source; use a new --out")
    else:
        env_path.write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    for seed in args.seeds:
        for method in args.methods:
            path = out/f"{method}-{seed}.json"
            if path.exists():
                print(f"verified protocol; resume skips {path.name}",flush=True)
                continue
            torch.manual_seed(seed)
            model = TinyDecoder(ModelConfig(method=method)).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
            g = torch.Generator().manual_seed(seed+10000)
            rng = random.Random(seed+20000)
            curve = []
            start = time.perf_counter()
            for step in range(args.steps):
                task = "lag4" if step%2==0 else "lookup"
                length = rng.choice([16,24,32] if task=="lag4" else [15,23,31])
                x,y = batch(task,args.batch_size,length,g,args.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[0][:,-1]
                loss = F.cross_entropy(logits,y)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                if (step+1)%100==0:
                    curve.append(dict(step=step+1,task=task,loss=loss.item()))
                    print(f"{method} seed={seed} step={step+1} loss={loss.item():.4f}",flush=True)
            metrics = []
            for task,lengths in [("lag4",[32,64,128]),("lookup",[31,63,127])]:
                for length in lengths:
                    metrics.append(evaluate(model,task,length,args.eval_batches,args.batch_size,args.device))
                    if method=="rope":
                        metrics.append(evaluate(model,task,length,args.eval_batches,args.batch_size,args.device,scale=4.0))
            data = dict(method=method,seed=seed,config=model.configuration(),
                parameters=sum(t.numel() for t in model.parameters()),source_sha256=provenance["source_sha256"],
                curve=curve,metrics=metrics,elapsed_seconds=time.perf_counter()-start)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(data,indent=2),encoding="utf-8")
            temp.replace(path)
            ckpt = ROOT/"experiments/checkpoints"
            ckpt.mkdir(parents=True,exist_ok=True)
            torch.save(dict(config=asdict(model.cfg),state_dict=model.state_dict()),ckpt/f"{method}-{seed}.pt")
            print(f"DONE {path.name}: {data['elapsed_seconds']:.1f}s",flush=True)

if __name__=="__main__":
    main()
