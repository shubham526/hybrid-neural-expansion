#!/bin/bash
"""
Modular NPRF Pipeline
Complete workflow using the refactored modular components
"""

set -e

echo "🚀 NPRF Modular Pipeline"
echo "========================"

# Configuration
DATA_DIR="data"
OUTPUT_DIR="nprf_output"
CONFIG="drmm_default"  # Use predefined config

# File paths
TRAIN_FILE="${DATA_DIR}/train.jsonl"
VAL_FILE="${DATA_DIR}/val.jsonl"
TEST_FILE="${DATA_DIR}/test.jsonl"

# Create output directories
mkdir -p "${OUTPUT_DIR}"

echo "📊 Configuration: ${CONFIG}"
echo "📁 Data directory: ${DATA_DIR}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo ""

# Check available configurations
echo "🔧 Available configurations:"
python nprf_main.py configs
echo ""

# Step 1: Training
echo "🎯 Step 1: Training NPRF Model"
echo "------------------------------"

if [ -f "${TRAIN_FILE}" ]; then
    python nprf_main.py train \
        --train-file "${TRAIN_FILE}" \
        --val-file "${VAL_FILE}" \
        --output-dir "${OUTPUT_DIR}/models" \
        --config "${CONFIG}"
    
    echo "✅ Training completed!"
else
    echo "❌ Training file not found: ${TRAIN_FILE}"
    echo "Please create training data first using create_train_test_data.py"
    exit 1
fi

echo ""

# Step 2: Inference
echo "🔍 Step 2: Running Inference"
echo "---------------------------"

if [ -f "${TEST_FILE}" ]; then
    TRAINED_MODEL="${OUTPUT_DIR}/models/nprf_drmm_model.pt"
    
    if [ -f "${TRAINED_MODEL}" ]; then
        echo "Using trained model: ${TRAINED_MODEL}"
        
        python nprf_main.py infer \
            --test-file "${TEST_FILE}" \
            --output-dir "${OUTPUT_DIR}/results" \
            --model-path "${TRAINED_MODEL}" \
            --config "${CONFIG}"
    else
        echo "⚠️  Trained model not found, using untrained model"
        
        python nprf_main.py infer \
            --test-file "${TEST_FILE}" \
            --output-dir "${OUTPUT_DIR}/results" \
            --config "${CONFIG}"
    fi
    
    echo "✅ Inference completed!"
else
    echo "❌ Test file not found: ${TEST_FILE}"
    exit 1
fi

echo ""

# Step 3: Results Summary
echo "📋 Step 3: Results Summary"
echo "------------------------"

RESULTS_FILE="${OUTPUT_DIR}/results/*.trec"
if ls ${RESULTS_FILE} 1> /dev/null 2>&1; then
    echo "Generated files:"
    ls -la "${OUTPUT_DIR}/results/"
    echo ""
    
    echo "Sample results (first 10 lines):"
    head -10 ${RESULTS_FILE}
    echo ""
    
    # Count results
    TOTAL_LINES=$(wc -l < ${RESULTS_FILE})
    UNIQUE_QUERIES=$(cut -d' ' -f1 ${RESULTS_FILE} | sort -u | wc -l)
    
    echo "📊 Statistics:"
    echo "  Total result lines: ${TOTAL_LINES}"
    echo "  Unique queries: ${UNIQUE_QUERIES}"
    echo "  Avg results per query: $((TOTAL_LINES / UNIQUE_QUERIES))"
    echo ""
else
    echo "❌ No results file found"
fi

echo "🎉 NPRF Modular Pipeline Completed!"
echo "=================================="
echo ""
echo "📁 Generated files:"
echo "  Models: ${OUTPUT_DIR}/models/"
echo "  Results: ${OUTPUT_DIR}/results/"
echo "  Config: ${OUTPUT_DIR}/models/nprf_config.json"
echo ""
echo "🔄 To run with different configurations:"
echo "  python nprf_main.py configs  # List available configs"
echo "  ./nprf_pipeline_modular.sh   # Edit CONFIG variable"
echo ""
echo "📈 To evaluate results:"
echo "  trec_eval qrels.txt ${RESULTS_FILE}"
echo ""
echo "🔧 To experiment:"
echo "  # Try different models"
echo "  python nprf_main.py train --config knrm_default ..."
echo "  python nprf_main.py train --config fast_training ..."
echo "  python nprf_main.py train --config high_quality ..."