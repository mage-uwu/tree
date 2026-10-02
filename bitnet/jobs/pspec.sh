# change-of-basis / regime-mixture attempts to rescue tree neuron scoring
python3 phase_treegate.py --layers 2,15 --trees 64 --cands 3072 --rot 64:3072 --mix 4:64:3072,8:64:3072,16:32:3072 --all "" --out /root/out/phase_spectral.jsonl
