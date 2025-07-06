"""
NPRF Data Module

Handles data loading, preprocessing, and dataset creation for NPRF.
"""

import torch
from torch.utils.data import Dataset
import random
import logging
from typing import List, Dict, Any, Tuple
from nprf_core import NPRFFeatureExtractor, load_jsonl

logger = logging.getLogger(__name__)


class NPRFDataset(Dataset):
    """Dataset for NPRF training and evaluation."""
    
    def __init__(self, data, feature_extractor, sample_size=10, mode='train'):
        self.data = data
        self.feature_extractor = feature_extractor
        self.sample_size = sample_size
        self.mode = mode
        
        if mode == 'train':
            self.pairs = self._generate_training_pairs()
        else:
            # For inference, we don't need pairs
            self.pairs = None
        
    def _generate_training_pairs(self):
        """Generate positive/negative pairs for training."""
        pairs = []
        
        for query_data in self.data:
            query_id = query_data['query_id']
            candidates = query_data['candidates']
            
            # Skip queries without enough candidates
            if len(candidates) < self.feature_extractor.nb_supervised_doc:
                logger.debug(f"Skipping query {query_id}: not enough candidates")
                continue
                
            # Group candidates by relevance
            relevant_docs = [c for c in candidates if c.get('relevance', 0) > 0]
            non_relevant_docs = [c for c in candidates if c.get('relevance', 0) == 0]
            
            if not relevant_docs or not non_relevant_docs:
                logger.debug(f"Skipping query {query_id}: no relevant or non-relevant docs")
                continue
                
            # Create pairs: (query_data, positive_doc, negative_doc)
            for pos_doc in relevant_docs:
                # Sample negative documents
                neg_samples = random.sample(
                    non_relevant_docs, 
                    min(self.sample_size, len(non_relevant_docs))
                )
                for neg_doc in neg_samples:
                    pairs.append((query_data, pos_doc, neg_doc))
                    
        logger.info(f"Generated {len(pairs)} training pairs")
        return pairs
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.pairs)
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self._get_training_item(idx)
        else:
            return self._get_inference_item(idx)
    
    def _get_training_item(self, idx):
        """Get training item (positive and negative features)."""
        query_data, pos_doc, neg_doc = self.pairs[idx]
        
        # Extract features for positive and negative documents
        pos_features = self._extract_features(query_data, pos_doc)
        neg_features = self._extract_features(query_data, neg_doc)
        
        return pos_features, neg_features
    
    def _get_inference_item(self, idx):
        """Get inference item (query data)."""
        return self.data[idx]
    
    def _extract_features(self, query_data, target_doc):
        """Extract features for a target document."""
        query_text = query_data['query_text']
        candidates = query_data['candidates']
        
        # Select PRF documents (top-k from first-stage retrieval)
        prf_docs = candidates[:self.feature_extractor.nb_supervised_doc]
        
        return self.feature_extractor.extract_features(query_text, target_doc, prf_docs)


class NPRFDataLoader:
    """Data loader factory for NPRF datasets."""
    
    @staticmethod
    def create_train_loader(data, feature_extractor, batch_size=20, sample_size=10, 
                           num_workers=4, shuffle=True):
        """Create training data loader."""
        dataset = NPRFDataset(data, feature_extractor, sample_size, mode='train')
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=NPRFDataLoader._collate_train_fn
        )
    
    @staticmethod
    def create_inference_loader(data, feature_extractor, batch_size=1, num_workers=1):
        """Create inference data loader."""
        dataset = NPRFDataset(data, feature_extractor, mode='inference')
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    @staticmethod
    def _collate_train_fn(batch):
        """Collate function for training data."""
        pos_features_list = []
        neg_features_list = []
        
        for pos_features, neg_features in batch:
            pos_features_list.append(pos_features)
            neg_features_list.append(neg_features)
        
        # Stack features
        if len(pos_features_list[0]) == 3:  # DRMM
            pos_dd_q = torch.stack([f[0] for f in pos_features_list])
            pos_dd_d = torch.stack([f[1] for f in pos_features_list])
            pos_doc_scores = torch.stack([f[2] for f in pos_features_list])
            
            neg_dd_q = torch.stack([f[0] for f in neg_features_list])
            neg_dd_d = torch.stack([f[1] for f in neg_features_list])
            neg_doc_scores = torch.stack([f[2] for f in neg_features_list])
            
            return (pos_dd_q, pos_dd_d, pos_doc_scores), (neg_dd_q, neg_dd_d, neg_doc_scores)
        
        else:  # K-NRM
            pos_dd = torch.stack([f[0] for f in pos_features_list])
            pos_doc_scores = torch.stack([f[1] for f in pos_features_list])
            
            neg_dd = torch.stack([f[0] for f in neg_features_list])
            neg_doc_scores = torch.stack([f[1] for f in neg_features_list])
            
            return (pos_dd, pos_doc_scores), (neg_dd, neg_doc_scores)


class DataPreprocessor:
    """Preprocesses data for NPRF training and inference."""
    
    @staticmethod
    def validate_data(data, min_candidates=5):
        """Validate and filter data."""
        valid_data = []
        
        for query_data in data:
            candidates = query_data.get('candidates', [])
            
            # Check minimum candidates
            if len(candidates) < min_candidates:
                logger.debug(f"Skipping query {query_data['query_id']}: too few candidates")
                continue
            
            # Check document text availability
            candidates_with_text = [c for c in candidates if 'doc_text' in c and c['doc_text'].strip()]
            
            if len(candidates_with_text) < min_candidates:
                logger.debug(f"Skipping query {query_data['query_id']}: too few candidates with text")
                continue
            
            # Update candidates to only include those with text
            query_data['candidates'] = candidates_with_text
            valid_data.append(query_data)
        
        logger.info(f"Validated data: {len(valid_data)}/{len(data)} queries retained")
        return valid_data
    
    @staticmethod
    def filter_by_relevance(data, require_positive=True, require_negative=True):
        """Filter data based on relevance requirements."""
        filtered_data = []
        
        for query_data in data:
            candidates = query_data['candidates']
            
            has_positive = any(c.get('relevance', 0) > 0 for c in candidates)
            has_negative = any(c.get('relevance', 0) == 0 for c in candidates)
            
            if require_positive and not has_positive:
                continue
            if require_negative and not has_negative:
                continue
                
            filtered_data.append(query_data)
        
        logger.info(f"Filtered by relevance: {len(filtered_data)}/{len(data)} queries retained")
        return filtered_data
    
    @staticmethod
    def balance_dataset(data, max_queries_per_class=None):
        """Balance dataset by number of relevant documents."""
        if max_queries_per_class is None:
            return data
        
        # Group by relevance level
        low_rel = []  # 0-2 relevant docs
        med_rel = []  # 3-10 relevant docs  
        high_rel = [] # 10+ relevant docs
        
        for query_data in data:
            candidates = query_data['candidates']
            num_relevant = sum(1 for c in candidates if c.get('relevance', 0) > 0)
            
            if num_relevant <= 2:
                low_rel.append(query_data)
            elif num_relevant <= 10:
                med_rel.append(query_data)
            else:
                high_rel.append(query_data)
        
        # Sample from each group
        balanced_data = []
        for group, name in [(low_rel, "low"), (med_rel, "medium"), (high_rel, "high")]:
            if len(group) > max_queries_per_class:
                sampled = random.sample(group, max_queries_per_class)
                logger.info(f"Sampled {len(sampled)} queries from {name} relevance group")
                balanced_data.extend(sampled)
            else:
                balanced_data.extend(group)
        
        random.shuffle(balanced_data)
        logger.info(f"Balanced dataset: {len(balanced_data)} total queries")
        return balanced_data


def load_and_preprocess_data(train_file, val_file=None, test_file=None, 
                           preprocess_config=None):
    """Load and preprocess all data files."""
    if preprocess_config is None:
        preprocess_config = {
            'min_candidates': 5,
            'require_positive': True,
            'require_negative': True,
            'balance_dataset': False,
            'max_queries_per_class': None
        }
    
    # Load data
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file) if val_file else None
    test_data = load_jsonl(test_file) if test_file else None
    
    # Preprocess each dataset
    datasets = {}
    
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        if data is None:
            continue
            
        logger.info(f"Preprocessing {name} data...")
        
        # Validate
        data = DataPreprocessor.validate_data(data, preprocess_config['min_candidates'])
        
        # Filter by relevance (only for train/val)
        if name in ['train', 'val']:
            data = DataPreprocessor.filter_by_relevance(
                data, 
                preprocess_config['require_positive'], 
                preprocess_config['require_negative']
            )
        
        # Balance dataset (only for training)
        if name == 'train' and preprocess_config['balance_dataset']:
            data = DataPreprocessor.balance_dataset(data, preprocess_config['max_queries_per_class'])
        
        datasets[name] = data
    
    return datasets
