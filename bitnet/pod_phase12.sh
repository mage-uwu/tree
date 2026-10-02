#!/bin/bash
# One-shot RunPod batch job (GPU pod, runpod/pytorch image): phase 1 baseline + phase 2 one-layer key sweep.
# Results are served read-only from /root/out on port 8888 (python -m http.server).
export DEBIAN_FRONTEND=noninteractive GIT_TERMINAL_PROMPT=0 PYTHONUNBUFFERED=1
mkdir -p /root/out; exec > >(tee -a /root/out/run.log) 2>&1
(cd /root/out && nohup python3 -m http.server 8888 >/dev/null 2>&1 &)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -m pip install -q "transformers==5.18.0" accelerate pyarrow 2>&1 | tail -1
python3 -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
cd /root && for i in 1 2 3 4 5; do rm -rf tree; git clone -q -b ${BRANCH:-claude/amazing-albattani-kor49n} https://github.com/mage-uwu/tree.git && break; sleep $((i*5)); done
cd /root/tree && git log --oneline -1 && cd bitnet
echo "=== phase12 START $(date -u +%T)"
python3 phase12.py --out /root/out/phase12.jsonl 2>&1 | grep -v -i "warning"
echo "=== ALL DONE $(date -u +%T)"
