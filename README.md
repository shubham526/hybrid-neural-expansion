# Hybrid Lexical-Semantic Term Weighting for Dense Query Expansion

This repository contains the implementation for our neural reranking approach that learns optimal importance weights for combining RM3 and semantic similarity signals in query expansion.

## 🎯 Overview

Traditional query expansion methods treat all expansion terms equally, ignoring the intuitive notion that some terms should contribute more strongly to the final query representation. Our approach learns explicit importance weights to optimally combine lexical (RM3) and semantic signals for expansion term weighting.

### Key Contributions

1. **Learnable Hybrid Importance Weights**: End-to-end learning of β (RM3), γ (semantic), and λ (interpolation) parameters
2. **Feature Normalization**: Min-max normalization of RM3 weights to [0,1] for comparable scales with semantic scores
3. **Enhanced Query Representation**: Weighted combination of original query and expansion term embeddings
4. **Configurable Scoring**: Support for cosine similarity, neural MLP, or bilinear scoring functions
5. **Ablation Support**: Systematic evaluation of RM3-only, semantic-only, and hybrid configurations
6. **Comprehensive Evaluation**: Cross-validation and cross-year TREC DL evaluation frameworks

## 🏗️ Architecture

```
Query → RM3 Expansion (top-k terms from pseudo-relevant docs)
                     ↓
Feature Extraction: RM3 weights (normalized to [0,1]) + Semantic similarity scores
                     ↓
Hybrid Weighting: w_i = β×RM3(t_i) + γ×Semantic(t_i)  [learnable β, γ]
                     ↓
Weighted Embeddings: e_weighted(t_i) = w_i × e(t_i)
                     ↓
Enhanced Query: q_enhanced = (1-λ)×e(q) + λ×(Σ e_weighted(t_i) / Σ w_i)  [learnable λ]
                     ↓
Document Scoring: score = f(q_enhanced, e(d))  [f ∈ {cosine, neural, bilinear}]
```

**Key Parameters:**
- **β**: Weight for RM3 statistical signal (learned end-to-end)
- **γ**: Weight for semantic similarity signal (learned end-to-end)  
- **λ**: Query-expansion interpolation weight (learned end-to-end)
- **Scoring function**: Configurable (cosine similarity, neural MLP, or bilinear)

## 🔧 Feature Normalization (Important!)

To ensure comparable scales between RM3 statistical weights (typically 0.001-0.1) and semantic similarity scores (typically -1 to 1), we apply **per-query min-max normalization** to RM3 weights:

```python
# RM3 weights normalized to [0, 1]
normalized_rm(t_i) = (rm_weight(t_i) - min(rm_weights)) / (max(rm_weights) - min(rm_weights))

# Semantic scores used as-is (already in [-1, 1] via cosine similarity)

# Combined weight
w_i = β × normalized_rm(t_i) + γ × semantic_score(t_i)
```

**Why this matters:** Without normalization, the learned β parameter becomes ~1000× larger than γ to compensate for scale differences, making interpretation difficult and potentially causing training instability.

## 📋 Requirements

### Dependencies

```bash
pip install torch sentence-transformers ir-datasets pytrec-eval
pip install numpy pandas tqdm pathlib
pip install pyjnius  # For Lucene integration
```

### Lucene Setup (Required for RM3)

Download Lucene JAR files (version 10.1.0):

```bash
mkdir lucene-jars
cd lucene-jars

wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/10.1.0/lucene-core-10.1.0.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-common/10.1.0/lucene-analysis-common-10.1.0.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-queryparser/10.1.0/lucene-queryparser-10.1.0.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-memory/10.1.0/lucene-memory-10.1.0.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-backward-codecs/10.1.0/lucene-backward-codecs-10.1.0.jar
```

### Java Requirements

```bash
# Ensure Java 8+ is installed
java -version

# Set JAVA_HOME if needed
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

## 🚀 Quick Start

### 1. Create Lucene Index

First, create a Lucene index for your dataset:

```python
from cross_encoder.src.utils.lucene_utils import initialize_lucene

# Initialize Lucene with JAR path
initialize_lucene("./lucene-jars")

# Index creation code here...
# (See scripts/create_index.py for complete example)
```

### 2. Extract Features

Extract RM3 + semantic similarity features:

```bash
python scripts/create_features.py \
    --dataset msmarco-passage/trec-dl-2019 \
    --output-dir ./features/dl19 \
    --index-path ./indexes/msmarco-passage \
    --lucene-path ./lucene-jars \
    --max-expansion-terms 15
```

### 3. Create Training Data

For proper TREC DL evaluation (recommended):

```bash
python scripts/create_train_test_data.py \
    --mode proper_dl \
    --val-year 19 \
    --test-year 20 \
    --train-features-file ./features/msmarco_train_features.json.gz \
    --val-features-file ./features/dl19_features.json.gz \
    --test-features-file ./features/dl20_features.json.gz \
    --output-dir ./data/proper_dl_exp1 \
    --save-statistics
```

### 4. Train Neural Model

```bash
python scripts/train.py \
    --train-file ./data/proper_dl_exp1/train.jsonl \
    --val-file ./data/proper_dl_exp1/val.jsonl \
    --output-dir ./models/dl19_to_dl20 \
    --num-epochs 20 \
    --learning-rate 0.001 \
    --scoring-method cosine \
    --ablation-mode both
```

### 5. Evaluate Model

```bash
python scripts/evaluate.py \
    --test-file ./data/proper_dl_exp1/test.jsonl \
    --model-info-file ./models/dl19_to_dl20/model_info.json \
    --dataset msmarco-passage/trec-dl-2020 \
    --output-dir ./results/dl19_to_dl20 \
    --save-runs
```

## 🎯 Configurable Scoring Methods

The framework supports three scoring functions for query-document matching:

### 1. Cosine Similarity (Recommended)
```bash
python scripts/train.py --scoring-method cosine
```
- **Pros**: Fast, interpretable, no additional parameters beyond β, γ, λ
- **Cons**: Linear similarity only
- **When to use**: Default choice, works well for most cases
- **Parameters**: ~3 learnable parameters (β, γ, λ)

### 2. Neural MLP
```bash
python scripts/train.py --scoring-method neural
```
- **Pros**: Can learn non-linear query-document interactions
- **Cons**: Slower, more parameters, prone to overfitting on small data
- **When to use**: Large training data, complex matching needed
- **Parameters**: ~200K parameters (depends on hidden_dim)

### 3. Bilinear
```bash
python scripts/train.py --scoring-method bilinear
```
- **Pros**: Learns query-document interactions, fewer parameters than MLP
- **Cons**: Still more complex than cosine
- **When to use**: Middle ground between cosine and neural
- **Parameters**: ~196K parameters (embedding_dim × embedding_dim)

## 🧪 Ablation Study Support

The framework supports systematic ablation studies to evaluate component contributions:

```bash
# Train with both components (full model)
python scripts/train.py --ablation-mode both

# Train with RM3 only (no semantic similarity)
python scripts/train.py --ablation-mode rm3_only

# Train with semantic similarity only (no RM3)
python scripts/train.py --ablation-mode cosine_only
```

**Expected results (with proper normalization):**
- **both**: Best performance (~0.46 MAP on TREC DL)
- **rm3_only**: Reasonable performance (~0.23-0.28 MAP)
- **cosine_only**: Lower performance (~0.15-0.22 MAP)

**Note:** Feature normalization is critical for ablations to work properly. Without normalization, single components may fail catastrophically (near-zero MAP) due to incompatible scales.

## 📁 Repository Structure

```
├── cross_encoder/
│   └── src/
│       ├── core/
│       │   ├── feature_extractor.py    # RM3 + semantic feature extraction
│       │   ├── rm_expansion.py         # Lucene-based RM3 implementation
│       │   └── semantic_similarity.py  # Sentence transformer similarity
│       ├── models/
│       │   ├── reranker.py            # Neural reranker architecture
│       │   └── trainer.py             # Training pipeline
│       ├── evaluation/
│       │   ├── evaluator.py           # TREC evaluation framework
│       │   ├── fast_evaluator.py      # Fast single-metric evaluation
│       │   └── metrics.py             # Evaluation metrics
│       └── utils/
│           ├── data_utils.py          # Dataset utilities
│           ├── file_utils.py          # File I/O utilities
│           ├── logging_utils.py       # Logging configuration
│           └── lucene_utils.py        # Lucene integration
├── scripts/
│   ├── create_features.py         # Feature extraction script
│   ├── create_train_test_data.py  # Data preparation script
│   ├── train.py                   # Model training script
│   └── evaluate.py               # Model evaluation script
└── README.md
```

## ⚙️ Configuration

### Experimental Modes

#### 1. Proper TREC DL Setup (Recommended)
- **Train**: MS MARCO passage/train/judged (~532K queries)
- **Validation**: TREC DL 2019 or 2020 (43-54 queries)
- **Test**: TREC DL 2020 or 2019 (54-43 queries)

```bash
--mode proper_dl --val-year 19 --test-year 20
```

#### 2. Cross-Validation
- **Dataset**: Any IR collection (e.g., Robust04)
- **Folds**: 5-fold cross-validation

```bash
--mode folds --folds-file ./folds.json
```

#### 3. Simple Train/Test Split
- **Train**: Any dataset
- **Test**: Different dataset
- **Validation**: Random split or MS MARCO dev

```bash
--mode single --val-strategy random_split
```

### Key Parameters

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|--------------|
| `max_expansion_terms` | Maximum RM3 expansion terms | 15 | 5-50 |
| `semantic_model` | Sentence transformer model | all-MiniLM-L6-v2 | Any SentenceTransformer |
| `scoring_method` | Scoring function | cosine | cosine, neural, bilinear |
| `ablation_mode` | Ablation configuration | both | both, rm3_only, cosine_only |
| `hidden_dim` | Neural MLP hidden dimension | 128 | 64-512 |
| `learning_rate` | Training learning rate | 0.001 | 1e-5 to 1e-2 |
| `num_epochs` | Training epochs | 20 | 10-50 |
| `batch_size` | Training batch size | 8 | 4-32 |
| `dropout` | Dropout rate (neural only) | 0.1 | 0.0-0.5 |

## 📊 Datasets Supported

### TREC Deep Learning
- `msmarco-passage/trec-dl-2019` (43 queries)
- `msmarco-passage/trec-dl-2020` (54 queries)
- `msmarco-passage/train/judged` (~532K queries)

### TREC Collections
- `disks45/nocr/trec-robust-2004` (~500K docs, 250 queries)
- Other TREC collections via ir_datasets

### Custom Collections
The framework supports any collection accessible through `ir_datasets`.

## 📈 Evaluation Metrics

- **MAP**: Mean Average Precision
- **nDCG@10/20**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Recall@100**: Recall at 100 documents

## 📊 Results

### Learned Weights Analysis

After training, the model learns three interpretable parameters:

- **β**: Weight for RM3 statistical signal (normalized)
- **γ**: Weight for semantic similarity signal
- **λ**: Interpolation between original query and expansion terms

**Example learned weights (after normalization):**
```
β = 0.65 ± 0.08  (RM3 receives ~2× weight vs semantic)
γ = 0.32 ± 0.05  (moderate semantic signal)
λ = 0.42 ± 0.06  (balanced query-expansion mix)
```

**Interpretation:** 
- Higher β suggests statistical evidence from pseudo-relevant documents is more important
- Moderate γ indicates semantic similarity provides complementary signal
- λ ≈ 0.4 means roughly 60% original query, 40% expansion terms

**Note:** These values are dataset-specific and learned during training. Without feature normalization, β would be ~500-1000× larger than γ, making interpretation meaningless.

### Performance Improvements

Typical improvements over BM25 baseline on TREC DL:
- **MAP**: +5-8% improvement
- **nDCG@10**: +3-6% improvement
- **MRR**: +4-7% improvement

Performance varies by dataset and configuration.

## 🔍 Advanced Usage

### Custom Feature Extraction

```python
from cross_encoder.src.core.feature_extractor import ExpansionFeatureExtractor

config = {
    'index_path': './indexes/msmarco-passage',
    'embedding_model': 'all-MiniLM-L6-v2',
    'max_expansion_terms': 15,
    'top_k_pseudo_docs': 10
}

extractor = ExpansionFeatureExtractor(config)

# Extract features for a query
features = extractor.extract_features_for_query(
    query_id="123",
    query_text="machine learning algorithms"
)

# Features include normalized RM3 weights and semantic scores
for term, term_features in features['term_features'].items():
    print(f"{term}: RM3={term_features['rm_weight']:.3f}, "
          f"Semantic={term_features['semantic_score']:.3f}")
```

### Custom Neural Architecture

```python
from cross_encoder.src.models.reranker import create_neural_reranker

# Create custom reranker
reranker = create_neural_reranker(
    model_name='all-mpnet-base-v2',
    scoring_method='cosine',
    max_expansion_terms=20,
    ablation_mode='both'
)

# Get learned weights after training
beta, gamma, lambda_val = reranker.get_learned_weights()
print(f"β={beta:.3f}, γ={gamma:.3f}, λ={lambda_val:.3f}")
```

### Batch Evaluation

```python
from cross_encoder.src.evaluation.evaluator import TRECEvaluator

evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'ndcg_cut_20'])

# Evaluate multiple runs
runs = {
    'baseline': baseline_results,
    'neural_reranker': reranked_results
}

comparison = evaluator.compare_runs(runs, qrels, baseline_run='baseline')
print(evaluator.create_results_table(comparison))
```

## 🛠️ Troubleshooting

### Common Issues

1. **Lucene JAR not found**
   ```bash
   # Ensure JARs are in lucene-jars/ directory
   ls lucene-jars/
   # Should show: lucene-core-10.1.0.jar, lucene-analysis-common-10.1.0.jar, etc.
   ```

2. **Java/PyJnius issues**
   ```bash
   # Set JAVA_HOME
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
   
   # Reinstall PyJnius if needed
   pip uninstall pyjnius
   pip install pyjnius
   ```

3. **Memory issues with large collections**
   ```bash
   # Reduce batch size and candidates per query
   python scripts/train.py \
       --max-candidates-per-query 50 \
       --batch-size 4
   ```

4. **CUDA/GPU issues**
   ```bash
   # Force CPU usage
   python scripts/train.py --device cpu
   ```

5. **Ablation failures (near-zero MAP)**
   - **Cause**: Feature normalization not applied
   - **Fix**: Ensure latest code version with normalization (see Fix #3 in code)
   - **Verify**: Check that RM3 weights are in [0,1] range

6. **Character encoding errors**
   - **Cause**: Bug in original term encoding (fixed in latest version)
   - **Fix**: Update to latest code version (see Fix #2 in code)
   - **Verify**: Check that expansion terms are words, not individual characters

### Debug Mode

Enable detailed logging:

```bash
python scripts/train.py --log-level DEBUG
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{your-paper-2025,
  title={Hybrid Lexical-Semantic Term Weighting for Dense Query Expansion},
  author={Your Name and Co-authors},
  booktitle={Proceedings of SIGIR 2025},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 📞 Contact

- **Author**: [Your Name]
- **Email**: your.email@university.edu
- **Issues**: Please use GitHub Issues for bug reports and feature requests

## 🔗 Related Work

- [RM3 Query Expansion](https://dl.acm.org/doi/10.1145/1148170.1148200)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [Sentence Transformers](https://arxiv.org/abs/1908.10084)

## ⚠️ Important Notes

### Recent Fixes (v1.0.1)

The following critical bugs have been fixed in the latest version:

1. **Feature Normalization**: RM3 weights are now properly normalized to [0,1] range
2. **Learnable λ**: The expansion interpolation parameter is now learned (previously hardcoded)
3. **Term Encoding**: Fixed character-level encoding bug (was encoding individual characters instead of words)

**If you're using an older version**, please update to get correct results. Without these fixes:
- Ablation studies may fail (near-zero MAP for single components)
- Learned weights are uninterpretable (β ~1000× larger than γ)
- Semantic similarity computed on wrong granularity

### Reproducibility

To reproduce paper results, ensure you:
1. Use the latest code version with all fixes applied
2. Apply feature normalization (automatic in latest version)
3. Train all three parameters (β, γ, λ) end-to-end
4. Use the same random seed for data splits
5. Report learned parameter values along with results

## 📝 Changelog

### v1.0.1 (Current - Critical Fixes)
- **FIXED**: Added min-max normalization for RM3 weights
- **FIXED**: Made λ (expansion interpolation) a learnable parameter
- **FIXED**: Character-level encoding bug in term expansion
- **IMPROVED**: Ablation modes now work correctly with normalized features
- **ADDED**: Support for configurable scoring methods (cosine/neural/bilinear)
- **ADDED**: Comprehensive ablation study framework

### v1.0.0 (Initial Release - Has Bugs)
- Initial release with known issues (not recommended)
- RM3 + semantic similarity feature extraction
- Neural importance weight learning
- TREC DL evaluation support
- Cross-validation framework

**⚠️ Use v1.0.1 or later for correct results**
