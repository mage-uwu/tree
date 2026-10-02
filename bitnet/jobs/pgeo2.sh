# phase 11b: stronger leaf distillation (lr 1e-3, 3200 steps) on storable configs, layer 15
python3 phase_geo.py --layer 15 --out /root/out/phase_geo2.jsonl --Ks 1,16 --ms 256,512,1024 --kl_cfgs "" --distill 1:1024,1:512,1:256,16:512 --lr 1e-3 --steps 3200
python3 phase_geo.py --layer 15 --out /root/out/phase_geo2.jsonl --Ks 1 --ms 1024 --kl_cfgs "" --distill 1:1024 --lr 3e-4 --steps 3200
