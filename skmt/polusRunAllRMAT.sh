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

bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph15CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_15.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph16CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_16.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph17CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_17.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph18CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_18.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph19CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_19.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph20CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_20.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph21CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_21.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph22CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_22.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph23CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_23.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph24CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_24.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/RMAT/graph25CSR $pr_flags -max_pr_iterations $maxPrIterations >> RMAT_25.log"