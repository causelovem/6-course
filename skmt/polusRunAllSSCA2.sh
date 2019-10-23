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

bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph16CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_16.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph17CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_17.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph18CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_18.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph19CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_19.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph20CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_20.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph21CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_21.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph22CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_22.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph23CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_23.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph24CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_24.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph25CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_25.log"
bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$pathToPR -num_gpus $numgpus -startwith $numgpus -graphfile ./generator/SSCA2/graph15CSR $pr_flags -max_pr_iterations $maxPrIterations >> SSCA2_15.log"