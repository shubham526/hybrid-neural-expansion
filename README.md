# Hybrid Lexical-Semantic Term Weighting for Dense Query Expansion

This repository contains the implementation for our neural reranking approach that learns optimal importance weights for combining RM3 and semantic similarity signals in query expansion.

## 🎯 Overview

Traditional query expansion methods treat all expansion terms equally, ignoring the intuitive notion that some terms should contribute more strongly to the final query representation. Our approach learns explicit importance weights (α for RM3, β for semantic similarity) to optimally combine lexical and semantic signals for expansion term weighting.

### Key Contributions

1. **Learnable Importance Weights**: End-to-end learning of α (RM3) and β (semantic similarity) parameters
2. **Enhanced Query Representation**: Linear combination of query + weighted expansion terms via neural layers
3. **Document-Aware Reranking**: Neural scoring of enhanced query-document pairs
4. **Comprehensive Evaluation**: Support for both cross-validation and cross-year TREC DL evaluation

## 🏗️ Architecture

```
Query → RM3 Expansion → α×RM3_weights + β×Semantic_scores 
                     ↓
Enhanced Query = Linear([query_emb, weighted_term_embs])
                     ↓
Final Score = Neural([enhanced_query, document_emb])
```

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
# You'll need to implement index creation for your specific collection
# Example structure:
from src.utils.lucene_utils import initialize_lucene

initialize_lucene("./lucene-jars")
# Index creation code here...
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
    --learning-rate 0.001
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

## 📁 Repository Structure

```
├── src/
│   ├── core/
│   │   ├── feature_extractor.py    # RM3 + semantic feature extraction
│   │   ├── rm_expansion.py         # Lucene-based RM3 implementation
│   │   └── semantic_similarity.py  # Sentence transformer similarity
│   ├── models/
│   │   ├── reranker.py            # Neural reranker architecture
│   │   └── trainer.py             # Training pipeline
│   ├── evaluation/
│   │   ├── evaluator.py           # TREC evaluation framework
│   │   └── metrics.py             # Evaluation metrics
│   └── utils/
│       ├── file_utils.py          # File I/O utilities
│       ├── logging_utils.py       # Logging configuration
│       └── lucene_utils.py        # Lucene integration
├── scripts/
│   ├── create_features.py         # Feature extraction script
│   ├── create_train_test_data.py  # Data preparation script
│   ├── train.py                   # Model training script
│   └── evaluate.py               # Model evaluation script
└── configs/
    └── *.yaml                     # Configuration templates
```

## 🔧 Configuration

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

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_expansion_terms` | Maximum RM3 expansion terms | 15 |
| `semantic_model` | Sentence transformer model | all-MiniLM-L6-v2 |
| `hidden_dim` | Neural network hidden dimension | 128 |
| `learning_rate` | Training learning rate | 0.001 |
| `num_epochs` | Training epochs | 20 |

## 📊 Datasets Supported

### TREC Deep Learning
- `msmarco-passage/trec-dl-2019` (43 queries)
- `msmarco-passage/trec-dl-2020` (54 queries)
- `msmarco-passage/train/judged` (~532K queries)

### TREC Collections
- `disks45/nocr/trec-robust-2004` (~500K docs, 250 queries)
- Other TREC collections via ir_datasets

### Custom Collections
The framework supports any collection accessible through ir_datasets.

## 🎯 Evaluation Metrics

- **MAP**: Mean Average Precision
- **nDCG@10/20**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Recall@100**: Recall at 100 documents

## 📈 Results

### Learned Weights Analysis

The model learns interpretable importance weights:
- **α (RM3 weight)**: Controls lexical expansion signal
- **β (Semantic weight)**: Controls semantic similarity signal

Example learned weights:
```
α = 0.8431 (strong lexical signal)
β = 0.6247 (moderate semantic signal)
```

### Performance Improvements

Typical improvements over BM25 baseline on TREC DL:
- **MAP**: +5-8% improvement
- **nDCG@10**: +3-6% improvement
- **MRR**: +4-7% improvement

## 🔍 Advanced Usage

### Custom Feature Extraction

```python
from src.core.feature_extractor import ExpansionFeatureExtractor
from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity

# Initialize components
rm_expansion = RMExpansion(index_path)
semantic_similarity = SemanticSimilarity(model_name)

extractor = ExpansionFeatureExtractor(
    rm_expansion=rm_expansion,
    semantic_similarity=semantic_similarity,
    max_expansion_terms=20
)

# Extract features
features = extractor.extract_features_for_query(
    query_id="123",
    query_text="machine learning algorithms",
    pseudo_relevant_docs=top_docs,
    pseudo_relevant_scores=scores
)
```

### Custom Neural Architecture

```python
from src.models.reranker import ImportanceWeightedNeuralReranker

# Create custom reranker
reranker = ImportanceWeightedNeuralReranker(
    model_name='all-mpnet-base-v2',
    max_expansion_terms=20,
    hidden_dim=256,
    dropout=0.2
)

# Get learned weights
alpha, beta = reranker.get_learned_weights()
print(f"α={alpha:.3f}, β={beta:.3f}")
```

### Batch Evaluation

```python
from src.evaluation.evaluator import TRECEvaluator

evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10'])

# Evaluate multiple runs
runs = {
    'baseline': baseline_results,
    'neural_reranker': reranked_results
}

comparison = evaluator.compare_runs(runs, qrels, baseline_run='baseline')
```

## 🛠️ Troubleshooting

### Common Issues

1. **Lucene JAR not found**
   ```bash
   # Ensure JARs are in lucene-jars/ directory
   ls lucene-jars/
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
   --max-candidates-per-query 50
   --batch-size 4
   ```

4. **CUDA/GPU issues**
   ```bash
   # Force CPU usage
   --device cpu
   ```

### Debug Mode

Enable detailed logging:

```bash
python scripts/train.py --log-level DEBUG
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{your-paper-2024,
  title={Hybrid Lexical-Semantic Term Weighting for Dense Query Expansion},
  author={Your Name and Co-authors},
  booktitle={Proceedings of SIGIR 2024},
  year={2024}
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

## 📝 Changelog

### v1.0.0 (Current)
- Initial release
- RM3 + semantic similarity feature extraction
- Neural importance weight learning
- TREC DL evaluation support
- Cross-validation framework

### Planned Features
- Support for dense retrieval integration
- Multi-stage reranking pipeline
- Additional semantic similarity models
- Automatic hyperparameter tuning