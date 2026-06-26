#!/bin/bash
# DGX Spark Deployment Script - Optimized for throughput
# Run on DGX Spark with 8x A100/H100 GPUs

set -e

# Configuration
NUM_WORKERS=8
CONFIG_FILE="config.yaml"
OUTPUT_DIR="pipeline_output_$(date +%Y%m%d_%H%M%S)"

# Generation mode (fast=2ns, high_quality=10ns)
MODE="${1:-fast}"  # Default to fast mode

echo "============================================================"
echo "  circRNA 3D Data Generation Pipeline - DGX Spark"
echo "============================================================"
echo "  Mode: $MODE"
echo "  Workers: $NUM_WORKERS"

# Environment setup
echo "Setting up environment..."

# Check for required packages
python -c "import RNA; print('ViennaRNA OK')" || echo "Install: conda install -c bioconda viennarna"
python -c "import openmm; print('OpenMM OK')" || echo "Install: pip install openmm"
python -c "import ray; print('Ray OK')" || echo "Install: pip install ray"

# RoseTTAFold2NA (install if not present)
if [ ! -d "RoseTTAFold2NA" ]; then
    echo "Cloning RoseTTAFold2NA..."
    git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git
    cd RoseTTAFold2NA
    # Follow RoseTTAFold2NA installation instructions
    cd ..
fi

# Step 1: Prefilter (optional but recommended for large batches)
if [ "$2" != "--skip-prefilter" ]; then
    echo ""
    echo "Step 1: Prefiltering sequences..."
    python prefilter.py --fasta "$2" --output prefiltered.fasta --config $CONFIG_FILE
    INPUT_FASTA="prefiltered.fasta"
else
    INPUT_FASTA="$2"
fi

# Time estimate based on mode
if [ "$MODE" == "fast" ]; then
    TIME_PER_SEQ="~1min"
    TOTAL_TIME="~2500min for 20k seqs (~42h with 8 workers)"
elif [ "$MODE" == "high_quality" ]; then
    TIME_PER_SEQ="~5min"
    TOTAL_TIME="~12500min for 20k seqs (~8.7 days with 8 workers)"
else
    TIME_PER_SEQ="~15sec"
    TOTAL_TIME="prefilter mode - no MD"
fi

echo ""
echo "Estimated time per sequence: $TIME_PER_SEQ"
echo "For 20,000 sequences: $TOTAL_TIME"
echo ""

# Run pipeline
echo "Starting pipeline..."
echo "Output directory: $OUTPUT_DIR"

python parallel_worker.py \
    --config $CONFIG_FILE \
    --fasta $INPUT_FASTA \
    --num-workers $NUM_WORKERS \
    --output $OUTPUT_DIR \
    --mode $MODE \
    --export-torusfold

echo ""
echo "============================================================"
echo "  Pipeline completed!"
echo "============================================================"
echo "  Results in: $OUTPUT_DIR"
echo "  TorusFold training data: $OUTPUT_DIR/torusfold_format/"
echo ""
echo "  Next steps:"
echo "    1. Check quality report in $OUTPUT_DIR/dataset_report.json"
echo "    2. Copy torusfold_format/ to TorusFold training data directory"
echo "    3. Run: python train_curriculum.py --labels $OUTPUT_DIR/torusfold_format"
