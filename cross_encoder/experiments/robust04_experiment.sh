#!/bin/bash
# TREC Robust 2004 5-Fold CV with Existing Folds File

set -e  # Exit on any error

echo "🚀 Starting TREC Robust 2004 5-Fold Cross-Validation Experiment"
echo "=================================================================="

# Configuration - UPDATE THESE PATHS
DATASET="disks45/nocr/trec-robust-2004"
FOLDS_FILE="./folds.json"  # YOUR EXISTING FOLDS FILE
BASE_DIR="./experiments/robust04_5fold"
INDEX_PATH="./indexes/robust04"  # YOUR ROBUST04 INDEX
LUCENE_PATH="./lucene-jars"

# Verify folds file exists
if [ ! -f "$FOLDS_FILE" ]; then
    echo "❌ Folds file not found: $FOLDS_FILE"
    echo "Please check the path to your folds.json file"
    exit 1
fi

echo "📁 Using existing folds file: $FOLDS_FILE"

# Create base directories
mkdir -p "$BASE_DIR"
mkdir -p "$BASE_DIR/features"
mkdir -p "$BASE_DIR/data"
mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/results"

# Check folds file content
echo "📊 Checking folds file content..."
python3 -c "
import json
with open('$FOLDS_FILE') as f:
    folds = json.load(f)
print(f'Found {len(folds)} folds:')
for fold_id, fold_data in folds.items():
    train_count = len(fold_data.get('training', []))
    test_count = len(fold_data.get('testing', []))
    print(f'  Fold {fold_id}: {train_count} train, {test_count} test')
"

# Step 1: Extract features (once for all folds)
echo ""
echo "🔍 Step 1: Extracting RM3 + Semantic features..."
python scripts/create_features.py \
    --dataset "$DATASET" \
    --output-dir "$BASE_DIR/features" \
    --index-path "$INDEX_PATH" \
    --lucene-path "$LUCENE_PATH" \
    --max-expansion-terms 15 \
    --semantic-model all-MiniLM-L6-v2 \
    --top-k-pseudo-docs 10 \
    --log-level INFO

echo "✅ Feature extraction completed"

# Step 2: Create train/test data for all folds
echo ""
echo "📝 Step 2: Creating train/test data for all folds..."
python scripts/create_train_test_data.py \
    --mode folds \
    --dataset "$DATASET" \
    --features-file "$BASE_DIR/features/disks45_nocr_trec-robust-2004_full_features.json.gz" \
    --folds-file "$FOLDS_FILE" \
    --output-dir "$BASE_DIR/data" \
    --max-candidates-per-query 100 \
    --ensure-positive-training \
    --save-statistics \
    --log-level INFO

echo "✅ Train/test data creation completed"

# Step 3: Train and evaluate each fold
echo ""
echo "🧠 Step 3: Training and evaluating all 5 folds..."

# Initialize results tracking
> "$BASE_DIR/fold_summary.txt"

for fold in 0 1 2 3 4; do
    echo ""
    echo "--- Processing Fold $fold ---"

    FOLD_DATA_DIR="$BASE_DIR/data/fold_$fold"
    FOLD_MODEL_DIR="$BASE_DIR/models/fold_$fold"
    FOLD_RESULTS_DIR="$BASE_DIR/results/fold_$fold"

    # Check if training data exists
    if [ ! -f "$FOLD_DATA_DIR/train.jsonl" ]; then
        echo "❌ Training data not found for fold $fold: $FOLD_DATA_DIR/train.jsonl"
        echo "Fold $fold: SKIPPED - No training data" >> "$BASE_DIR/fold_summary.txt"
        continue
    fi

    # Create fold directories
    mkdir -p "$FOLD_MODEL_DIR"
    mkdir -p "$FOLD_RESULTS_DIR"

    # Check training data size
    TRAIN_SIZE=$(wc -l < "$FOLD_DATA_DIR/train.jsonl")
    TEST_SIZE=$(wc -l < "$FOLD_DATA_DIR/test.jsonl")
    echo "📊 Fold $fold: $TRAIN_SIZE training examples, $TEST_SIZE test examples"

    echo "🏋️ Training model for fold $fold..."
    python scripts/train.py \
        --train-file "$FOLD_DATA_DIR/train.jsonl" \
        --output-dir "$FOLD_MODEL_DIR" \
        --model-name all-MiniLM-L6-v2 \
        --max-expansion-terms 15 \
        --hidden-dim 128 \
        --num-epochs 20 \
        --learning-rate 0.001 \
        --weight-decay 0.0001 \
        --training-mode pointwise \
        --loss-type mse \
        --log-level INFO

    if [ $? -eq 0 ]; then
        echo "✅ Training completed for fold $fold"

        # Extract and log learned weights
        if [ -f "$FOLD_MODEL_DIR/model_info.json" ]; then
            WEIGHTS=$(python3 -c "
import json
model_info = json.load(open('$FOLD_MODEL_DIR/model_info.json'))
weights = model_info['learned_weights']
print(f'α={weights[\"alpha\"]:.4f}, β={weights[\"beta\"]:.4f}')
")
            echo "📊 Fold $fold learned weights: $WEIGHTS"
            echo "Fold $fold: TRAINED - Weights: $WEIGHTS" >> "$BASE_DIR/fold_summary.txt"
        fi

        echo "🔬 Evaluating model for fold $fold..."
        python scripts/evaluate.py \
            --test-file "$FOLD_DATA_DIR/test.jsonl" \
            --model-info-file "$FOLD_MODEL_DIR/model_info.json" \
            --dataset "$DATASET" \
            --output-dir "$FOLD_RESULTS_DIR" \
            --save-runs \
            --top-k 100 \
            --log-level INFO

        if [ $? -eq 0 ]; then
            echo "✅ Evaluation completed for fold $fold"

            # Extract key metrics
            if [ -f "$FOLD_RESULTS_DIR/evaluation_results.json" ]; then
                METRICS=$(python3 -c "
import json
results = json.load(open('$FOLD_RESULTS_DIR/evaluation_results.json'))
if 'neural_reranker' in results:
    metrics = results['neural_reranker']
    map_score = metrics.get('map', 0)
    ndcg10 = metrics.get('ndcg_cut_10', 0)
    mrr = metrics.get('recip_rank', 0)
    print(f'MAP={map_score:.4f}, nDCG@10={ndcg10:.4f}, MRR={mrr:.4f}')
else:
    print('No neural_reranker results found')
")
                echo "📈 Fold $fold results: $METRICS"
                echo "Fold $fold: EVALUATED - $METRICS" >> "$BASE_DIR/fold_summary.txt"
            fi
        else
            echo "❌ Evaluation failed for fold $fold"
            echo "Fold $fold: EVAL_FAILED" >> "$BASE_DIR/fold_summary.txt"
        fi
    else
        echo "❌ Training failed for fold $fold"
        echo "Fold $fold: TRAIN_FAILED" >> "$BASE_DIR/fold_summary.txt"
    fi

    echo "--- Fold $fold completed ---"
done

# Step 4: Aggregate results across all folds
echo ""
echo "📊 Step 4: Aggregating results across all folds..."

python3 -c "
import json
import numpy as np
from pathlib import Path

base_dir = Path('$BASE_DIR')
results_dir = base_dir / 'results'

# Collect results from all folds
fold_results = {}
all_learned_weights = {}
baseline_results = {}
metrics = ['map', 'ndcg_cut_10', 'ndcg_cut_20', 'recip_rank', 'recall_100']

print('\\n' + '=' * 60)
print('INDIVIDUAL FOLD RESULTS')
print('=' * 60)

for fold in range(5):
    fold_results_file = results_dir / f'fold_{fold}' / 'evaluation_results.json'
    fold_model_file = base_dir / 'models' / f'fold_{fold}' / 'model_info.json'

    if fold_results_file.exists():
        with open(fold_results_file) as f:
            fold_result = json.load(f)
            fold_results[f'fold_{fold}'] = fold_result

        if fold_model_file.exists():
            with open(fold_model_file) as f:
                model_info = json.load(f)
                all_learned_weights[f'fold_{fold}'] = model_info['learned_weights']

        print(f'\\nFold {fold}:')

        # Neural reranker results
        if 'neural_reranker' in fold_result:
            print('  Neural Reranker:')
            for metric in metrics:
                if metric in fold_result['neural_reranker']:
                    score = fold_result['neural_reranker'][metric]
                    print(f'    {metric}: {score:.4f}')

        # Baseline results
        if 'baseline' in fold_result:
            print('  Baseline:')
            for metric in metrics:
                if metric in fold_result['baseline']:
                    score = fold_result['baseline'][metric]
                    print(f'    {metric}: {score:.4f}')

        # Learned weights
        if f'fold_{fold}' in all_learned_weights:
            weights = all_learned_weights[f'fold_{fold}']
            print(f'  Learned weights: α={weights[\"alpha\"]:.4f}, β={weights[\"beta\"]:.4f}')

# Aggregate neural reranker metrics across folds
print('\\n' + '=' * 60)
print('AGGREGATED RESULTS (NEURAL RERANKER)')
print('=' * 60)

aggregated_results = {
    'experiment': 'TREC Robust 2004 5-Fold CV',
    'num_folds': len(fold_results),
    'fold_results': fold_results,
    'learned_weights': all_learned_weights,
    'neural_reranker_metrics': {},
    'baseline_metrics': {}
}

# Aggregate neural reranker results
print('\\nNeural Reranker Performance:')
for metric in metrics:
    neural_values = []
    baseline_values = []

    for fold_id, results in fold_results.items():
        if 'neural_reranker' in results and metric in results['neural_reranker']:
            neural_values.append(results['neural_reranker'][metric])
        if 'baseline' in results and metric in results['baseline']:
            baseline_values.append(results['baseline'][metric])

    if neural_values:
        aggregated_results['neural_reranker_metrics'][metric] = {
            'mean': float(np.mean(neural_values)),
            'std': float(np.std(neural_values)),
            'values': neural_values,
            'num_folds': len(neural_values)
        }
        print(f'  {metric.upper():>12}: {np.mean(neural_values):.4f} ± {np.std(neural_values):.4f}')

    if baseline_values:
        aggregated_results['baseline_metrics'][metric] = {
            'mean': float(np.mean(baseline_values)),
            'std': float(np.std(baseline_values)),
            'values': baseline_values,
            'num_folds': len(baseline_values)
        }

# Show baseline performance
if aggregated_results['baseline_metrics']:
    print('\\nBaseline Performance:')
    for metric in metrics:
        if metric in aggregated_results['baseline_metrics']:
            stats = aggregated_results['baseline_metrics'][metric]
            print(f'  {metric.upper():>12}: {stats[\"mean\"]:.4f} ± {stats[\"std\"]:.4f}')

# Calculate improvements
print('\\nImprovement over Baseline:')
for metric in metrics:
    if (metric in aggregated_results['neural_reranker_metrics'] and
        metric in aggregated_results['baseline_metrics']):
        neural_mean = aggregated_results['neural_reranker_metrics'][metric]['mean']
        baseline_mean = aggregated_results['baseline_metrics'][metric]['mean']
        improvement_pct = ((neural_mean - baseline_mean) / baseline_mean) * 100
        print(f'  {metric.upper():>12}: {improvement_pct:+.2f}%')

# Aggregate learned weights
if all_learned_weights:
    alpha_values = [w['alpha'] for w in all_learned_weights.values()]
    beta_values = [w['beta'] for w in all_learned_weights.values()]

    aggregated_results['aggregated_weights'] = {
        'alpha': {
            'mean': float(np.mean(alpha_values)),
            'std': float(np.std(alpha_values)),
            'values': alpha_values
        },
        'beta': {
            'mean': float(np.mean(beta_values)),
            'std': float(np.std(beta_values)),
            'values': beta_values
        }
    }

    print('\\n' + '=' * 60)
    print('LEARNED IMPORTANCE WEIGHTS')
    print('=' * 60)
    print(f'α (RM3 weight):      {np.mean(alpha_values):.4f} ± {np.std(alpha_values):.4f}')
    print(f'β (Semantic weight): {np.mean(beta_values):.4f} ± {np.std(beta_values):.4f}')

    print('\\nPer-fold weights:')
    for fold_id, weights in all_learned_weights.items():
        print(f'  {fold_id}: α={weights[\"alpha\"]:.4f}, β={weights[\"beta\"]:.4f}')

# Save aggregated results
aggregated_file = base_dir / 'aggregated_results.json'
with open(aggregated_file, 'w') as f:
    json.dump(aggregated_results, f, indent=2)

print(f'\\n✅ Saved aggregated results to: {aggregated_file}')
print('\\n' + '=' * 60)
print('5-FOLD CROSS-VALIDATION COMPLETED!')
print('=' * 60)
print(f'📁 Results directory: {base_dir}')
print(f'📊 Aggregated results: {aggregated_file}')
print(f'🏆 Individual fold results: {base_dir}/results/fold_*/evaluation_results.json')
print('=' * 60)
"

# Show final summary
echo ""
echo "📋 Fold Summary:"
cat "$BASE_DIR/fold_summary.txt"

echo ""
echo "🎯 EXPERIMENT COMPLETED!"
echo "========================"
echo "📁 Results directory: $BASE_DIR"
echo "📊 Aggregated results: $BASE_DIR/aggregated_results.json"
echo "📋 Fold summary: $BASE_DIR/fold_summary.txt"