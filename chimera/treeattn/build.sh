#!/bin/sh
# Builds the C engine and the micro-benchmarks. Needs gcc and an x86-64 CPU with AVX2 + FMA + F16C.
set -e
cd "$(dirname "$0")"
gcc -O3 -march=native -Wno-unused-result -Wno-misleading-indentation -o treeattn treeattn.c -lm
gcc -O3 -march=native -o scanbench scanbench.c -lm
gcc -O3 -march=native -o mlpbench mlpbench.c -lm
gcc -O3 -march=native -ffast-math -o attnbench attnbench.c -lm
echo "built: treeattn scanbench mlpbench attnbench"
