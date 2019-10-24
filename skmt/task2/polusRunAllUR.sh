# Number of GPUs
numgpus=1

# Optimization flag
pr_flags="-noopt"

# Path to executable PageRank
pathToPR=./code/groute/build/pr

# Maximum number of iterations of pagerank
maxPrIterations=50

# Dumping factor (ALPHA) are set in pr_common.h (using define). Change it and rerun setup.sh
# Epsilon of algortihm (EPSILON) are set in pr_common.h (using define). Change it and rerun setup.sh

# Run without printing output and checking results

bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph16CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log15"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph15CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log16"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph17CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log17"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph18CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log18"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph19CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log19"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph20CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log20"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph21CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log21"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph22CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log22"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph23CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log23"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph24CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log24"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/UR/graph25CSR $pr_flags -max_pr_iterations $maxPrIterations >> UR.log25"