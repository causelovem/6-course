# Number of GPUs
numgpus=1

# Path to graph file
# pathToGraph=./graphCSR/rmatCSR
pathToGraph=(./graphCSR/rmat17CSR ./graphCSR/rmat18CSR ./graphCSR/rmat19CSR ./graphCSR/rmat20CSR ./graphCSR/rmat21CSR ./graphCSR/rmat22CSR ./graphCSR/rmat23CSR ./graphCSR/rmat24CSR ./graphCSR/rmat25CSR ./graphCSR/rmat26CSR)

# Path to executable PageRank
SSSP=./code/groute/build/sssp

# Path to output ranks
pathToOutput=res

# Path to logs
pathToLogs=groute-sssp.log

# Run with printing output and checking results
# $SSSP -num_gpus $numgpus -startwith $numgpus --prio_delta=$sssp_prio_delta_fused \
#  -graphfile $fullpath -output /tmp/sssptmp.txt 2>&1
for n in 17 18 19 20 21 22 23 24 25 26
do
	bsub -q normal -o output -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" "$SSSP -num_gpus $numgpus -startwith $numgpus -graphfile ./graphCSR/rmat${n}CSR -output ${pathToOutput}${n} -check -verbose >> $pathToLogs"
done
# $SSSP -num_gpus $numgpus -startwith $numgpus -graphfile $pathToGraph -output $pathToOutput -check -verbose >> $pathToLogs
