"""
NPRF Core Module

Contains shared components for NPRF training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SimilarityComputer:
    """Compute similarity matrices between documents using embeddings."""
    
    def __init__(self, model_name="bert-base-uncased", device=None, max_length=128):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def get_embeddings(self, texts: List[str]):
        """Get embeddings for texts."""
        if not texts:
            return torch.zeros((0, self.model.config.hidden_size))
            
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings.cpu()
    
    def compute_similarity_matrix(self, doc1_text: str, doc2_text: str, max_words=50):
        """Compute word-level similarity matrix between two documents."""
        doc1_words = doc1_text.lower().split()[:max_words]
        doc2_words = doc2_text.lower().split()[:max_words]
        
        if not doc1_words or not doc2_words:
            return torch.zeros((1, 1))
            
        doc1_embeds = self.get_embeddings(doc1_words)
        doc2_embeds = self.get_embeddings(doc2_words)
        
        doc1_norm = F.normalize(doc1_embeds, p=2, dim=1)
        doc2_norm = F.normalize(doc2_embeds, p=2, dim=1)
        
        similarity_matrix = torch.mm(doc1_norm, doc2_norm.t())
        return similarity_matrix


class HistogramFeatures:
    """Convert similarity matrices to histogram features for DRMM."""
    
    def __init__(self, hist_size=30):
        self.hist_size = hist_size
        
    def compute_histogram(self, sim_matrix: torch.Tensor, query_length: int):
        """Convert similarity matrix to histogram features."""
        if sim_matrix.numel() == 0:
            return torch.zeros((query_length, self.hist_size))
            
        hist = torch.zeros((query_length, self.hist_size))
        
        for i in range(min(query_length, sim_matrix.shape[0])):
            row = sim_matrix[i, :]
            for val in row:
                bin_idx = int((val + 1.0) / 2.0 * (self.hist_size - 1))
                bin_idx = max(0, min(bin_idx, self.hist_size - 1))
                hist[i, bin_idx] += 1
                
        hist = torch.log(hist + 1.0)
        return hist


class KernelFeatures:
    """Convert similarity matrices to kernel features for K-NRM."""
    
    def __init__(self, kernel_size=11, lambda_val=0.5):
        self.kernel_size = kernel_size
        self.lambda_val = lambda_val
        self.mu_list = self._get_mu_list()
        self.sigma_list = self._get_sigma_list()
        
    def _get_mu_list(self):
        """Get mu values for Gaussian kernels."""
        mu_list = [1.0]  # Exact match
        if self.kernel_size == 1:
            return mu_list
            
        bin_size = 2.0 / (self.kernel_size - 1)
        mu_list.append(1 - bin_size / 2)
        for i in range(1, self.kernel_size - 1):
            mu_list.append(mu_list[i] - bin_size)
        return mu_list
    
    def _get_sigma_list(self):
        """Get sigma values for Gaussian kernels."""
        bin_size = 2.0 / (self.kernel_size - 1)
        sigma_list = [0.00001]  # Exact match
        if self.kernel_size == 1:
            return sigma_list
        sigma_list.extend([bin_size * self.lambda_val] * (self.kernel_size - 1))
        return sigma_list
    
    def compute_kernel_features(self, sim_matrix: torch.Tensor):
        """Convert similarity matrix to kernel features."""
        if sim_matrix.numel() == 0:
            return torch.zeros(self.kernel_size)
            
        kernel_features = torch.zeros(self.kernel_size)
        
        for i, (mu, sigma) in enumerate(zip(self.mu_list, self.sigma_list)):
            kernel_val = torch.exp(-torch.square(sim_matrix - mu) / (2 * sigma * sigma))
            kde = torch.sum(kernel_val, dim=1)
            kde = torch.log(torch.clamp(kde, min=1e-10)) * 0.01
            kernel_features[i] = torch.sum(kde)
            
        return kernel_features


class NPRFDRMMModel(nn.Module):
    """NPRF model using DRMM as the underlying neural IR model."""
    
    def __init__(self, hist_size=30, hidden_size=5, nb_supervised_doc=10, doc_topk_term=20):
        super().__init__()
        self.hist_size = hist_size
        self.hidden_size = hidden_size
        self.nb_supervised_doc = nb_supervised_doc
        self.doc_topk_term = doc_topk_term
        
        # Query term gating
        self.query_gate = nn.Linear(1, 1, bias=False)
        
        # Document interaction modeling
        self.hidden_layers = nn.Sequential(
            nn.Linear(hist_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Document weighting
        self.doc_gate = nn.Linear(1, 1, bias=False)
        
        # Final scoring
        self.final_layer = nn.Linear(nb_supervised_doc, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights as in original implementation."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gate' in name:
                    nn.init.uniform_(param, -0.01, 0.01)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
        
    def forward(self, dd_q_input, dd_d_input, doc_scores):
        """Forward pass."""
        # Query term gating
        q_weights = torch.softmax(self.query_gate(dd_q_input), dim=2)
        
        # Document interaction
        d_scores = self.hidden_layers(dd_d_input)
        
        # Combine query and document signals
        combined = torch.sum(q_weights * d_scores, dim=2)
        
        # Document weighting
        doc_weights = self.doc_gate(doc_scores).squeeze(-1)
        weighted_scores = combined * doc_weights
        
        # Final scoring
        final_score = self.final_layer(weighted_scores)
        
        return final_score


class NPRFKNRMModel(nn.Module):
    """NPRF model using K-NRM as the underlying neural IR model."""
    
    def __init__(self, kernel_size=11, hidden_size=11, nb_supervised_doc=10):
        super().__init__()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.nb_supervised_doc = nb_supervised_doc
        
        # Document interaction modeling
        self.hidden_layer = nn.Sequential(
            nn.Linear(kernel_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Document weighting
        self.doc_gate = nn.Linear(1, 1, bias=False)
        
        # Final scoring
        self.final_layer = nn.Linear(nb_supervised_doc, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights as in original implementation."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gate' in name:
                    nn.init.uniform_(param, -0.01, 0.01)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
        
    def forward(self, dd_input, doc_scores):
        """Forward pass."""
        # Document interaction
        d_scores = self.hidden_layer(dd_input).squeeze(-1)
        
        # Document weighting
        doc_weights = self.doc_gate(doc_scores).squeeze(-1)
        weighted_scores = d_scores * doc_weights
        
        # Final scoring
        final_score = self.final_layer(weighted_scores)
        
        return final_score


class NPRFFeatureExtractor:
    """Unified feature extractor for NPRF models."""
    
    def __init__(self, model_type, similarity_computer, nb_supervised_doc=10, 
                 doc_topk_term=20, hist_size=30, kernel_size=11):
        self.model_type = model_type.lower()
        self.similarity_computer = similarity_computer
        self.nb_supervised_doc = nb_supervised_doc
        self.doc_topk_term = doc_topk_term
        
        if self.model_type == "drmm":
            self.feature_extractor = HistogramFeatures(hist_size)
        else:  # knrm
            self.feature_extractor = KernelFeatures(kernel_size)
    
    def extract_features(self, query_text: str, target_doc: Dict[str, Any], 
                        prf_docs: List[Dict[str, Any]]) -> Tuple[torch.Tensor, ...]:
        """Extract features for a target document given PRF documents."""
        if self.model_type == "drmm":
            return self._extract_drmm_features(query_text, target_doc, prf_docs)
        else:
            return self._extract_knrm_features(query_text, target_doc, prf_docs)
    
    def _extract_drmm_features(self, query_text, target_doc, prf_docs):
        """Extract DRMM features."""
        query_terms = query_text.lower().split()[:self.doc_topk_term]
        
        dd_q_feat = torch.zeros((self.nb_supervised_doc, self.doc_topk_term, 1))
        dd_d_feat = torch.zeros((self.nb_supervised_doc, self.doc_topk_term, self.feature_extractor.hist_size))
        doc_scores = torch.zeros((self.nb_supervised_doc, 1))
        
        if 'doc_text' not in target_doc:
            return dd_q_feat, dd_d_feat, doc_scores
            
        target_text = target_doc['doc_text']
        
        for i, prf_doc in enumerate(prf_docs[:self.nb_supervised_doc]):
            if 'doc_text' not in prf_doc:
                continue
                
            prf_text = prf_doc['doc_text']
            
            # Compute similarity matrix
            sim_matrix = self.similarity_computer.compute_similarity_matrix(prf_text, target_text)
            
            # Convert to histogram features
            hist_features = self.feature_extractor.compute_histogram(sim_matrix, self.doc_topk_term)
            dd_d_feat[i, :hist_features.shape[0], :] = hist_features
            
            # Query term importance
            dd_q_feat[i, :len(query_terms), 0] = 1.0
            
            # Document scores
            doc_scores[i, 0] = prf_doc['score']
        
        # Normalize document scores
        doc_scores = self._normalize_scores(doc_scores)
        return dd_q_feat, dd_d_feat, doc_scores
    
    def _extract_knrm_features(self, query_text, target_doc, prf_docs):
        """Extract K-NRM features."""
        dd_feat = torch.zeros((self.nb_supervised_doc, self.feature_extractor.kernel_size))
        doc_scores = torch.zeros((self.nb_supervised_doc, 1))
        
        if 'doc_text' not in target_doc:
            return dd_feat, doc_scores
            
        target_text = target_doc['doc_text']
        
        for i, prf_doc in enumerate(prf_docs[:self.nb_supervised_doc]):
            if 'doc_text' not in prf_doc:
                continue
                
            prf_text = prf_doc['doc_text']
            
            # Compute similarity matrix
            sim_matrix = self.similarity_computer.compute_similarity_matrix(prf_text, target_text)
            
            # Convert to kernel features
            kernel_features = self.feature_extractor.compute_kernel_features(sim_matrix)
            dd_feat[i, :] = kernel_features
            
            # Document scores
            doc_scores[i, 0] = prf_doc['score']
        
        # Normalize document scores
        doc_scores = self._normalize_scores(doc_scores)
        return dd_feat, doc_scores
    
    def _normalize_scores(self, doc_scores):
        """Normalize document scores to [0.5, 1.0] range."""
        if doc_scores.max() > doc_scores.min():
            return 0.5 * (doc_scores - doc_scores.min()) / (doc_scores.max() - doc_scores.min()) + 0.5
        else:
            doc_scores.fill_(0.75)
            return doc_scores


def load_jsonl(file_path):
    """Load JSONL data."""
    import json
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_trec_run(all_results, output_file, run_name):
    """Write results in TREC run format."""
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(f"{result['query_id']} Q0 {result['doc_id']} {result['rank']} {result['score']} {run_name}\n")


def hinge_loss(pos_scores, neg_scores, margin=1.0):
    """Compute hinge loss for ranking."""
    return torch.mean(torch.clamp(margin - pos_scores + neg_scores, min=0))


class ModelFactory:
    """Factory for creating NPRF models."""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """Create NPRF model based on type."""
        if model_type.lower() == "drmm":
            return NPRFDRMMModel(
                hist_size=kwargs.get('hist_size', 30),
                hidden_size=kwargs.get('hidden_size', 5),
                nb_supervised_doc=kwargs.get('nb_supervised_doc', 10),
                doc_topk_term=kwargs.get('doc_topk_term', 20)
            )
        elif model_type.lower() == "knrm":
            return NPRFKNRMModel(
                kernel_size=kwargs.get('kernel_size', 11),
                hidden_size=kwargs.get('hidden_size', 11),
                nb_supervised_doc=kwargs.get('nb_supervised_doc', 10)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def load_model(model_path, device=None):
        """Load trained model from checkpoint."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        model_type = checkpoint['model_type']
        args = checkpoint['args']
        
        model = ModelFactory.create_model(model_type, **args)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, checkpoint
