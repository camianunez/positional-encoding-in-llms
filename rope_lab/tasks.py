"""Online synthetic next-token tasks: answers never enter the input."""
import torch
VALUES, BOS, LAG4, ASK, KEY0, KEYS, VOCAB = 16,16,17,18,20,64,84

def batch(task,size,length,generator,device="cpu"):
    """length counts input tokens and excludes the held-out answer."""
    if task=="lag4":
        if length<6:
            raise ValueError("lag4 needs at least six input tokens")
        x = torch.randint(VALUES,(size,length),generator=generator)
        x[:,0],x[:,-1] = BOS,LAG4
        y = x[:,-5].clone()
    elif task=="lookup":
        if length<5 or length%2!=1 or length>2*KEYS+3:
            raise ValueError("lookup requires odd length and 1..64 complete pairs")
        pairs = (length-3)//2
        keys = torch.rand(size,KEYS,generator=generator).argsort(dim=-1)[:,:pairs]
        values = torch.randint(VALUES,(size,pairs),generator=generator)
        chosen = torch.randint(pairs,(size,),generator=generator)
        x = torch.empty(size,length,dtype=torch.long)
        x[:,0] = BOS
        x[:,1:-2:2],x[:,2:-2:2] = keys+KEY0,values
        x[:,-2],x[:,-1] = ASK,keys[torch.arange(size),chosen]+KEY0
        y = values[torch.arange(size),chosen]
    else:
        raise ValueError(f"unknown task {task}")
    return x.to(device),y.to(device)
