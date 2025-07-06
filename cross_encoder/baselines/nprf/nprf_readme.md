# NPRF: Neural Pseudo Relevance Feedback (Modular Implementation)

Clean, modular PyTorch implementation of the NPRF framework from "NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval" (EMNLP 2018).

## 🏗️ Modular Architecture

The codebase is now organized into clean, reusable modules:

```
nprf_core.py          # Core models and feature extraction
nprf_data.py          # Data loading and preprocessing  
nprf_config.py        # Configuration management
nprf_trainer_modular.py    # Training pipeline
nprf_reranker_modular.py   # Inference pipeline
nprf_main.py          # Unified entry point
```

### Key Benefits
- **No Code Duplication**: Shared components across training and inference
- **Configuration Management**: Predefined configs for common scenarios
- **Clean Interfaces**: Modular design with clear separation of concerns
- **Easy Extension**: Simple to add new models or features

## 🚀 Quick Start

### 1. Unified Interface

Use the main entry point for all operations:

```bash
# List available configurations
python nprf_main.py configs

# Train with predefined config
python nprf_main.py train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --output-dir models/ \
  --config drmm_default

# Run inference
python nprf_main.py infer \
  --test-file data/test.jsonl \
  --model-path models/nprf_drmm_model.pt \
  --output-dir results/ \
  --config drmm_default
```

### 2. Complete Pipeline

Run the full pipeline with one command:

```bash
./nprf_pipeline_modular.sh
```

## 📋 Available Configurations

| Config | Model | Use Case | Training Time |
|--------|-------|----------|---------------|
| `drmm_default` | DRMM | General purpose | ~30 min |
| `knrm_default` | K-NRM | Faster training | ~20 min |
| `fast_training` | DRMM | Quick experiments | ~10 min |
| `high_quality` | DRMM | Best performance | ~60 min |

### Configuration Details

```python
# View configuration
from nprf_config import get_config
config = get_config("drmm_default")
print(config.model)  # Model hyperparameters
print(config.training)  # Training settings
```

## 🎯 Training

### Basic Training

```bash
python nprf_main.py train \
  --train-file train.jsonl \
  --output-dir models/ \
  --config drmm_default
```

### Custom Training

```bash
python nprf_main.py train \
  --train-file train.jsonl \
  --val-file val.jsonl \
  --output-dir models/ \
  --model-type drmm \
  --num-epochs 50 \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --nb-supervised-doc 15
```

### Advanced Options

```bash
# Balance dataset by relevance
python nprf_main.py train \
  --train-file train.jsonl \
  --balance-dataset \
  --max-queries-per-class 100

# Fast training for debugging
python nprf_main.py train \
  --train-file train.jsonl \
  --config fast_training
```

## 🔍 Inference

### Basic Inference

```bash
python nprf_main.py infer \
  --test-file test.jsonl \
  --model-path models/nprf_drmm_model.pt \
  --output-dir results/
```

### Batch Processing

```bash
# Multi-threaded inference
python nprf_main.py infer \
  --test-file large_test.jsonl \
  --model-path trained_model.pt \
  --output-dir results/ \
  --max-workers 8
```

### Score Combination

```bash
# Adjust original vs NPRF score weighting
python nprf_main.py infer \
  --test-file test.jsonl \
  --model-path model.pt \
  --score-combination-weight 0.8  # 80% original, 20% NPRF
```

## 📊 Configuration System

### Using Predefined Configs

```python
from nprf_config import get_config

# Load configuration
config = get_config("drmm_default")

# Modify if needed
config.training.num_epochs = 50
config.model.nb_supervised_doc = 15

# Save custom config
config.save_json("my_config.json")
```

### Creating Custom Configs

```python
from nprf_config import NPRFConfig, ModelConfig, TrainingConfig

config = NPRFConfig(
    model=ModelConfig(
        model_type="drmm",
        nb_supervised_doc=20,
        hist_size=40
    ),
    training=TrainingConfig(
        num_epochs=40,
        learning_rate=0.0005
    )
)
```

## 🧪 Experimentation

### Model Comparison

```bash
# Train both models
python nprf_main.py train --config drmm_default --output-dir models/drmm/
python nprf_main.py train --config knrm_default --output-dir models/knrm/

# Compare results
python nprf_main.py infer --model-path models/drmm/nprf_drmm_model.pt --run-name drmm
python nprf_main.py infer --model-path models/knrm/nprf_knrm_model.pt --run-name knrm
```

### Hyperparameter Search

```bash
# Try different PRF depths
for depth in 5 10 15 20; do
  python nprf_main.py train \
    --nb-supervised-doc $depth \
    --output-dir models/depth_$depth/
done
```

### Ablation Studies

```bash
# Test score combination weights
for weight in 0.5 0.6 0.7 0.8 0.9; do
  python nprf_main.py infer \
    --score-combination-weight $weight \
    --run-name "nprf_w${weight}"
done
```

## 🔧 Extension Points

### Adding New Models

```python
# In nprf_core.py
class NPRFNewModel(nn.Module):
    def __init__(self, ...):
        # Custom architecture
        pass
    
    def forward(self, ...):
        # Custom forward pass
        pass

# In ModelFactory
@staticmethod
def create_model(model_type, **kwargs):
    if model_type == "new_model":
        return NPRFNewModel(**kwargs)
```

### Custom Feature Extractors

```python
class CustomFeatureExtractor:
    def extract_features(self, query_text, target_doc, prf_docs):
        # Custom feature extraction
        pass
```

### New Configurations

```python
# In nprf_config.py
CONFIGS["my_config"] = NPRFConfig(
    model=ModelConfig(...),
    training=TrainingConfig(...),
    # ...
)
```

## 📈 Performance Tips

### GPU Optimization
- Use larger batch sizes: `--batch-size 64`
- Increase workers: `--num-workers 8`
- Mixed precision training (implement torch.cuda.amp)

### Memory Optimization
- Reduce document terms: `--doc-topk-term 10`
- Smaller PRF depth: `--nb-supervised-doc 5`
- Lower histogram size: `--hist-size 20`

### Speed Optimization
- Use fast config: `--config fast_training`
- Fewer epochs: `--num-epochs 10`
- Single worker inference: `--max-workers 1`

## 🐛 Troubleshooting

### Common Issues

**Training Loss Not Decreasing**
```bash
# Try lower learning rate
python nprf_main.py train --learning-rate 0.0001

# Check data balance
python nprf_main.py train --balance-dataset
```

**Out of Memory**
```bash
# Reduce batch size
python nprf_main.py train --batch-size 8

# Use fast config
python nprf_main.py train --config fast_training
```

**Poor Results**
```bash
# Ensure model is trained
ls models/nprf_*_model.pt

# Check score combination
python nprf_main.py infer --score-combination-weight 0.9
```

## 📦 Module Dependencies

```python
# Core dependencies
torch>=1.9.0
transformers>=4.0.0
numpy>=1.19.0
scipy>=1.6.0

# Optional for evaluation
scikit-learn>=0.24.0
```

## 🏃‍♂️ Development Workflow

```bash
# 1. Quick test with fast config
python nprf_main.py train --config fast_training

# 2. Full training
python nprf_main.py train --config drmm_default

# 3. Evaluate
python nprf_main.py infer --model-path models/nprf_drmm_model.pt

# 4. Compare configurations
python nprf_main.py configs
```

This modular implementation eliminates redundancy while providing a clean, extensible foundation for NPRF research and development.

## 📚 API Reference

### Core Classes

#### `NPRFFeatureExtractor`
```python
extractor = NPRFFeatureExtractor(
    model_type="drmm",           # "drmm" or "knrm"
    similarity_computer=sim_comp, # SimilarityComputer instance
    nb_supervised_doc=10,        # Number of PRF documents
    doc_topk_term=20,           # Terms per document
    hist_size=30,               # Histogram bins (DRMM)
    kernel_size=11              # Kernel count (K-NRM)
)

# Extract features for a document
features = extractor.extract_features(query_text, target_doc, prf_docs)
```

#### `NPRFTrainer`
```python
trainer = NPRFTrainer(model, feature_extractor, device)

# Train model
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=30,
    learning_rate=0.001
)

# Save trained model
trainer.save_model(output_path, args, feature_extractor)
```

#### `NPRFReranker`
```python
reranker = NPRFReranker(
    model_path="path/to/model.pt",  # Optional: path to trained model
    model_type="drmm",              # For untrained models
    device=device
)

# Rerank query candidates
results = reranker.rerank_query(query_data)
```

### Configuration Management

#### `NPRFConfig`
```python
from nprf_config import NPRFConfig, ModelConfig, TrainingConfig

# Create configuration
config = NPRFConfig(
    model=ModelConfig(model_type="drmm", nb_supervised_doc=10),
    training=TrainingConfig(num_epochs=30, batch_size=20)
)

# Save/load configuration
config.save_json("config.json")
loaded_config = NPRFConfig.from_json("config.json")

# Use predefined configurations
config = get_config("drmm_default")
```

### Data Loading

#### `NPRFDataLoader`
```python
# Training data loader
train_loader = NPRFDataLoader.create_train_loader(
    data=train_data,
    feature_extractor=extractor,
    batch_size=20,
    sample_size=10,
    num_workers=4
)

# Inference data loader  
infer_loader = NPRFDataLoader.create_inference_loader(
    data=test_data,
    feature_extractor=extractor,
    batch_size=1
)
```

## 🔬 Research Extensions

### 1. New Similarity Functions

```python
class SemanticSimilarityComputer(SimilarityComputer):
    """Use sentence transformers for document similarity."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def compute_similarity_matrix(self, doc1_text, doc2_text):
        # Sentence-level similarity instead of word-level
        doc1_sents = doc1_text.split('.')[:10]
        doc2_sents = doc2_text.split('.')[:10]
        
        doc1_embeds = self.model.encode(doc1_sents)
        doc2_embeds = self.model.encode(doc2_sents)
        
        similarity = torch.tensor(np.dot(doc1_embeds, doc2_embeds.T))
        return similarity
```

### 2. Attention-Based Features

```python
class AttentionFeatures:
    """Attention-based feature extraction."""
    
    def __init__(self, embed_dim=768):
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
    
    def compute_attention_features(self, query_embeds, doc_embeds):
        # Cross-attention between query and document
        attn_output, attn_weights = self.attention(
            query_embeds, doc_embeds, doc_embeds
        )
        return attn_output, attn_weights
```

### 3. Graph-Based PRF

```python
class GraphNPRF(nn.Module):
    """Graph neural network for PRF document relationships."""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.gcn = GraphConvolution(hidden_dim, hidden_dim)
        
    def forward(self, doc_features, adjacency_matrix):
        # Model relationships between PRF documents
        graph_features = self.gcn(doc_features, adjacency_matrix)
        return graph_features
```

### 4. Multi-Stage Reranking

```python
class MultiStageNPRF:
    """Multi-stage reranking with different PRF depths."""
    
    def __init__(self, stages=[5, 10, 20]):
        self.stages = stages
        self.rerankers = [NPRFReranker(prf_depth=s) for s in stages]
    
    def rerank_multistage(self, query_data):
        candidates = query_data['candidates']
        
        for stage, reranker in zip(self.stages, self.rerankers):
            # Rerank with increasing PRF depth
            candidates = reranker.rerank_query({
                **query_data,
                'candidates': candidates[:stage*2]
            })
        
        return candidates
```

## 🎯 Evaluation Framework

### Comprehensive Evaluation

```python
class NPRFEvaluator:
    """Comprehensive evaluation framework."""
    
    def __init__(self, metrics=['map', 'ndcg', 'mrr']):
        self.metrics = metrics
    
    def evaluate_model(self, model_path, test_data, qrels):
        # Load model and run inference
        reranker = NPRFReranker(model_path=model_path)
        
        results = []
        for query_data in test_data:
            query_results = reranker.rerank_query(query_data)
            results.extend(query_results)
        
        # Compute metrics
        return self.compute_metrics(results, qrels)
    
    def compute_metrics(self, results, qrels):
        # Implementation of IR metrics
        pass
```

### Cross-Validation Framework

```python
def run_cross_validation(data, config, k_folds=5):
    """Run k-fold cross-validation."""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        # Train model for this fold
        trainer = NPRFTrainer(...)
        model = trainer.train(train_data, val_data)
        
        # Evaluate on validation set
        evaluator = NPRFEvaluator()
        metrics = evaluator.evaluate_model(model, val_data, qrels)
        
        fold_results.append(metrics)
        logger.info(f"Fold {fold}: {metrics}")
    
    return fold_results
```

## 🚀 Production Deployment

### Batch Processing Script

```python
#!/usr/bin/env python3
"""
Production NPRF batch processing.
"""

import argparse
from pathlib import Path
from nprf_reranker_modular import NPRFReranker

def batch_rerank(input_dir, output_dir, model_path, batch_size=1000):
    """Process large datasets in batches."""
    
    reranker = NPRFReranker(model_path=model_path)
    input_files = list(Path(input_dir).glob("*.jsonl"))
    
    for input_file in input_files:
        logger.info(f"Processing {input_file}")
        
        # Load data in batches
        all_data = load_jsonl(input_file)
        
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i+batch_size]
            
            # Process batch
            batch_results = []
            for query_data in batch:
                results = reranker.rerank_query(query_data)
                batch_results.extend(results)
            
            # Save batch results
            output_file = output_dir / f"{input_file.stem}_batch_{i//batch_size}.trec"
            write_trec_run(batch_results, output_file, "nprf_batch")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True) 
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    
    args = parser.parse_args()
    batch_rerank(args.input_dir, args.output_dir, args.model_path, args.batch_size)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy NPRF modules
COPY nprf_*.py ./
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "nprf_api.py"]
```

### REST API Server

```python
# nprf_api.py
from flask import Flask, request, jsonify
from nprf_reranker_modular import NPRFReranker

app = Flask(__name__)
reranker = NPRFReranker(model_path="models/nprf_drmm_model.pt")

@app.route('/rerank', methods=['POST'])
def rerank_endpoint():
    """REST endpoint for reranking."""
    query_data = request.json
    
    try:
        results = reranker.rerank_query(query_data)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

## 📖 Citation

```bibtex
@inproceedings{li2018nprf,
  title={NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval},
  author={Li, Canjia and Sun, Yingfei and He, Ben and Wang, Le and Hui, Kai and Yates, Andrew and Sun, Le and Xu, Jungang},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={4482--4491},
  year={2018}
}
```# NPRF: Neural Pseudo Relevance Feedback for Information Retrieval

PyTorch implementation of the NPRF framework from the paper "NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval" (EMNLP 2018).

## Overview

NPRF enhances neural information retrieval models by incorporating pseudo relevance feedback through document-to-document interactions. It uses top-ranked documents from first-stage retrieval as "expansion queries" to better score target documents.

### Key Features

- **Two Neural Variants**: DRMM and K-NRM based models
- **Document-to-Document Matching**: Uses PRF documents to enhance target document scoring
- **End-to-End Training**: Neural components learn optimal interaction patterns
- **PyTorch Implementation**: Modern, efficient implementation with GPU support

## Requirements

```bash
pip install torch transformers scipy scikit-learn numpy
```

## Quick Start

### 1. Prepare Your Data

First, create train/test data using the provided data creation script:

```bash
python create_train_test_data.py \
  --mode single \
  --train-dataset "msmarco-passage/train/judged" \
  --test-dataset "msmarco-passage/trec-dl-2020" \
  --train-features-file features_train.json \
  --test-features-file features_test.json \
  --output-dir data/
```

This creates JSONL files where each example contains:
- Query text
- List of candidate documents from first-stage retrieval
- Relevance labels
- Document text content

### 2. Train NPRF Model

```bash
python nprf_trainer.py \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --output-dir models/ \
  --model-type drmm \
  --num-epochs 30 \
  --batch-size 20
```

**Key Arguments:**
- `--model-type`: Choose `drmm` or `knrm`
- `--nb-supervised-doc`: Number of PRF documents (default: 10)
- `--sample-size`: Negative samples per positive (default: 10)
- `--learning-rate`: Adam learning rate (default: 0.001)

### 3. Run Inference

```bash
python nprf_reranker.py \
  --test-file data/test.jsonl \
  --model-path models/nprf_drmm_model.pt \
  --output-dir results/ \
  --run-name nprf_drmm
```

This generates TREC-format run files for evaluation.

## Model Architecture

### NPRF-DRMM

1. **Similarity Computation**: Computes word-level similarity matrices between PRF documents and target documents
2. **Histogram Features**: Converts similarities to histogram features for each query term
3. **Query Term Gating**: Learns importance weights for query terms
4. **Document Weighting**: Uses first-stage scores to weight PRF documents
5. **Final Scoring**: Combines all signals for final relevance score

### NPRF-K-NRM

1. **Kernel Features**: Uses Gaussian kernels to capture different similarity levels
2. **Document Interaction**: Models interactions between PRF docs and targets
3. **Document Weighting**: Incorporates first-stage retrieval confidence
4. **Neural Combination**: Learns optimal feature combination

## Training Process

The training follows these steps:

1. **Pair Generation**: Creates positive/negative document pairs from relevance judgments
2. **Feature Extraction**: 
   - Selects top-k documents as PRF documents
   - Computes document-to-document similarity matrices
   - Converts to neural features (histograms or kernels)
3. **Neural Training**: Uses hinge loss for ranking optimization
4. **Validation**: Monitors performance on held-out queries

### Training Data Format

Each training example in JSONL format:

```json
{
  "query_id": "1234",
  "query_text": "what is machine learning",
  "candidates": [
    {
      "doc_id": "doc1",
      "score": 15.2,
      "relevance": 2,
      "doc_text": "Machine learning is..."
    }
  ]
}
```

## Advanced Usage

### Cross-Validation Training

```bash
# Train multiple folds
for fold in {1..5}; do
  python nprf_trainer.py \
    --train-file data/fold_${fold}/train.jsonl \
    --val-file data/fold_${fold}/val.jsonl \
    --output-dir models/fold_${fold}/ \
    --model-type drmm
done
```

### Hyperparameter Tuning

Key hyperparameters to experiment with:

- `--nb-supervised-doc`: Number of PRF documents (5-20)
- `--doc-topk-term`: Terms per document (10-50)
- `--hist-size`: Histogram bins for DRMM (20-50)
- `--kernel-size`: Number of kernels for K-NRM (7-15)
- `--hidden-size`: Neural layer size (5-20)

### Batch Processing

For large datasets, use batch processing:

```bash
python nprf_reranker.py \
  --test-file large_test.jsonl \
  --model-path trained_model.pt \
  --output-dir results/ \
  --max-workers 8 \
  --batch-size 32
```

## Evaluation

Evaluate results using standard IR metrics:

```bash
# Using trec_eval
trec_eval qrels.txt results/nprf_drmm.trec

# Common metrics: MAP, P@10, NDCG@10, MRR
trec_eval -m map -m P.10 -m ndcg_cut.10 qrels.txt results/nprf_drmm.trec
```

## Performance Tips

### GPU Usage
- Models automatically use GPU if available
- Use `CUDA_VISIBLE_DEVICES=0` to specify GPU
- Increase batch size for better GPU utilization

### Memory Optimization
- Reduce `--doc-topk-term` for memory savings
- Lower `--max-workers` if running out of memory
- Use smaller embeddings models if needed

### Speed Optimization
- Cache similarity computations for repeated queries
- Use mixed precision training: `--fp16`
- Increase `--num-workers` for data loading

## Architecture Comparison

| Component | NPRF-DRMM | NPRF-K-NRM |
|-----------|-----------|------------|
| Features | Histograms | Kernels |
| Query Modeling | Term-level gating | Document-level |
| Similarity | Binned counts | Gaussian weights |
| Training Speed | Slower | Faster |
| Memory Usage | Higher | Lower |

## Common Issues

### Training Issues

**Low convergence**: Try reducing learning rate or increasing PRF depth
**Memory errors**: Reduce batch size or document length limits
**No improvement**: Check data quality and positive/negative balance

### Inference Issues

**Slow inference**: Reduce similarity computation complexity
**Poor results**: Ensure model was trained on similar data distribution
**Missing documents**: Check document text availability in test data

## Extending the Framework

### Custom Similarity Functions

```python
class CustomSimilarityComputer(SimilarityComputer):
    def compute_similarity_matrix(self, doc1_text, doc2_text):
        # Custom similarity computation
        pass
```

### New Neural Models

```python
class NPRFCustomModel(nn.Module):
    def __init__(self, ...):
        # Custom architecture
        pass
    
    def forward(self, features, doc_scores):
        # Custom forward pass
        pass
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{li2018nprf,
  title={NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval},
  author={Li, Canjia and Sun, Yingfei and He, Ben and Wang, Le and Hui, Kai and Yates, Andrew and Sun, Le and Xu, Jungang},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year={2018}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper and repository for licensing terms.