gcc matrix_cpu.c -o matrix_cpu -O2
# command to compile the CPU matrix multiplication program

N_VALUES=(256 512 1024 2048 4096)

OUTPUT_DIR="../results"
OUTPUT_FILE="$OUTPUT_DIR/cpu-results.csv"

mkdir -p $OUTPUT_DIR

# CSV header
echo "N,time_seconds" > $OUTPUT_FILE

echo "Running CPU matrix multiplication benchmarks..."

for N in "${N_VALUES[@]}"; do
    echo "Running N=$N"
    OUTPUT=$(./matrix_cpu $N)
    TIME=$(echo $OUTPUT | awk '{print $5}')
    echo "$N,$TIME" >> $OUTPUT_FILE
done

echo "Done. Results saved to $OUTPUT_FILE"
