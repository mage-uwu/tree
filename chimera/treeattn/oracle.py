import torch, torch.nn.functional as F
from tree_attn import LM
import data
torch.set_num_threads(2)
ck=torch.load('runs/full_shakespeare.pt'); m=LM(ck['cfg']); m.load_state_dict(ck['state']); m.eval()
get,_,_=data.shakespeare(128)
def run(k):
    g=torch.Generator().manual_seed(123); tot=0; cov=0
    with torch.no_grad():
        for _ in range(10):
            x,y=get('val',32,g); h=m.emb(x)
            for l,(n,a) in enumerate(zip(m.norms,m.attn)):
                h=h+a(n(h)); u=m.n2[l](h); gact=F.gelu(m.up[l](u)); imp=gact.abs()*m.down[l].weight.abs().sum(0)
                top=imp.topk(k,-1).indices; mask=torch.zeros_like(gact).scatter_(-1,top,1.)
                cov+= ((imp*mask).sum(-1)/imp.sum(-1)).mean().item()
                h=h+m.down[l](gact*mask)
            tot+=F.cross_entropy(m.head(m.norm(h)).flatten(0,1),y.flatten()).item()
    return tot/10, cov/40
for k in [16,32,64,128,256]: print("oracle per-token top-%d: loss %.4f  importance covered %.2f"%(k,*run(k)))
