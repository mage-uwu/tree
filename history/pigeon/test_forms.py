import torch
from attn_model import AttnLM
torch.manual_seed(0)
for kinds, rd, loc in [("RRRR",0,0),("RRRR",4,0),("RRRR",4,1),("SRRR",4,1),("SSSS",0,0)]:
    cfg=dict(vocab=65,d=128,layers=1,kinds=kinds,rdepth=rd,local=bool(loc))
    m=AttnLM(cfg).double().eval()
    x=torch.randint(65,(2,100))
    with torch.no_grad():
        par=m(x,"parallel"); chk=m(x,"chunk",chunk=16)
        st=m.init_state(2,dt=torch.float64); rec=[]
        for t in range(100):
            l,st=m.step(x[:,t],st); rec.append(l)
        rec=torch.stack(rec,1)
    print(f"{kinds} depth={rd} local={loc}: |par-chunk| {(par-chk).abs().max():.2e}  |par-rec| {(par-rec).abs().max():.2e}")
    if rd:
        P,_=m.mixers[0].router(m.mixers[0]._heads(m.mixers[0].k(m.norms[0](m.emb(x)))))
        print("   leaves used by keys (head1):", (P[:,1].sum((0,1))>0).sum().item(), "/", P.shape[-1])
