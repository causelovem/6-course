if [ ! -n "$1" ]
then
    echo "I need type of graph"
    exit -1
fi

# Number of GPUs
numgpus=1

# Path to graph file
# pathToGraph=./graphCSR/rmatCSR
# pathToGraph=(./graphCSR/rmat17CSR ./graphCSR/rmat18CSR ./graphCSR/rmat19CSR ./graphCSR/rmat20CSR ./graphCSR/rmat21CSR ./graphCSR/rmat22CSR ./graphCSR/rmat23CSR ./graphCSR/rmat24CSR ./graphCSR/rmat25CSR)

# Path to executable SSSP
SSSP=./code/groute/build/sssp

# Path to output res
pathToOutput=res

# Path to logs
pathToLogs=sssp.log

# Run with printing output and checking results
# $SSSP -num_gpus $numgpus -startwith $numgpus --prio_delta=$sssp_prio_delta_fused \
#  -graphfile $fullpath -output /tmp/sssptmp.txt 2>&1
for n in 17 18 19 20 21 22 23 24 25
do
    bsub -q normal -o output$1${n} -gpu "num=2:mode=exclusive_process:mps=yes" -n 1 -R "affinity[core(20,exclusive=(socket,alljobs))]" \
    "$SSSP -num_gpus $numgpus -startwith $numgpus -graphfile ./graphCSR/$1${n}CSR -output ./res/${pathToOutput}$1${n} -check -verbose >> ./logs/$1${n}${pathToLogs}"
done
# $SSSP -num_gpus $numgpus -startwith $numgpus -graphfile $pathToGraph -output $pathToOutput -check -verbose >> $pathToLogs
