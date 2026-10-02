# phase 11: MLP in neuron space (clusters by active neurons, routed exact subsets, distillation); layers 15, 2, 25
python3 phase_geo.py --layer 15 --out /root/out/phase_geo.jsonl
python3 phase_geo.py --layer 2  --out /root/out/phase_geo.jsonl --Ks 1,16,64 --kl_cfgs 16:1024,64:1024 --distill 1:1024,64:1024
python3 phase_geo.py --layer 25 --out /root/out/phase_geo.jsonl --Ks 1,16,64 --kl_cfgs 16:1024,64:1024 --distill 1:1024,64:1024
