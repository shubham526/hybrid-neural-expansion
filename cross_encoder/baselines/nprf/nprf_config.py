"""
NPRF Configuration Module

Centralized configuration management for NPRF experiments.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for NPRF models."""
    model_type: str = "drmm"  # "drmm" or "knrm"
    nb_supervised_doc: int = 10
    doc_topk_term: int = 20
    hist_size: int = 30  # For DRMM
    kernel_size: int = 11  # For K-NRM
    hidden_size: int = 5
    
    def __post_init__(self):
        if self.model_type not in ["drmm", "knrm"]:
            raise ValueError(f"Invalid model_type: {self.model_type}")


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_epochs: int = 30
    batch_size: int = 20
    learning_rate: float = 0.001
    sample_size: int = 10
    num_workers: int = 4
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    min_candidates: int = 5
    require_positive: bool = True
    require_negative: bool = True
    balance_dataset: bool = False
    max_queries_per_class: Optional[int] = None


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    score_combination_weight: float = 0.7
    max_workers: int = 2
    run_name: str = "nprf"


@dataclass
class NPRFConfig:
    """Complete NPRF configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    inference: InferenceConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            inference=InferenceConfig(**config_dict.get('inference', {}))
        )
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'inference': self.inference.__dict__
        }
    
    def save_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined configurations
CONFIGS = {
    "drmm_default": NPRFConfig(
        model=ModelConfig(
            model_type="drmm",
            nb_supervised_doc=10,
            doc_topk_term=20,
            hist_size=30,
            hidden_size=5
        ),
        training=TrainingConfig(
            num_epochs=30,
            batch_size=20,
            learning_rate=0.001,
            sample_size=10
        ),
        data=DataConfig(
            min_candidates=5,
            require_positive=True,
            require_negative=True
        ),
        inference=InferenceConfig(
            score_combination_weight=0.7,
            run_name="nprf_drmm"
        )
    ),
    
    "knrm_default": NPRFConfig(
        model=ModelConfig(
            model_type="knrm",
            nb_supervised_doc=10,
            kernel_size=11,
            hidden_size=11
        ),
        training=TrainingConfig(
            num_epochs=30,
            batch_size=20,
            learning_rate=0.001,
            sample_size=10
        ),
        data=DataConfig(
            min_candidates=5,
            require_positive=True,
            require_negative=True
        ),
        inference=InferenceConfig(
            score_combination_weight=0.7,
            run_name="nprf_knrm"
        )
    ),
    
    "fast_training": NPRFConfig(
        model=ModelConfig(
            model_type="drmm",
            nb_supervised_doc=5,
            doc_topk_term=10,
            hist_size=20,
            hidden_size=3
        ),
        training=TrainingConfig(
            num_epochs=10,
            batch_size=32,
            learning_rate=0.002,
            sample_size=5
        ),
        data=DataConfig(
            min_candidates=3,
            balance_dataset=True,
            max_queries_per_class=100
        ),
        inference=InferenceConfig(
            score_combination_weight=0.8,
            run_name="nprf_fast"
        )
    ),
    
    "high_quality": NPRFConfig(
        model=ModelConfig(
            model_type="drmm",
            nb_supervised_doc=15,
            doc_topk_term=30,
            hist_size=50,
            hidden_size=10
        ),
        training=TrainingConfig(
            num_epochs=50,
            batch_size=16,
            learning_rate=0.0005,
            sample_size=15
        ),
        data=DataConfig(
            min_candidates=10,
            require_positive=True,
            require_negative=True
        ),
        inference=InferenceConfig(
            score_combination_weight=0.6,
            run_name="nprf_hq"
        )
    )
}


def get_config(config_name: str) -> NPRFConfig:
    """Get predefined configuration."""
    if config_name not in CONFIGS:
        available = list(CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")
    return CONFIGS[config_name]


def list_configs():
    """List available configurations."""
    return list(CONFIGS.keys())


def create_config_from_args(args) -> NPRFConfig:
    """Create configuration from command line arguments."""
    model_config = ModelConfig(
        model_type=getattr(args, 'model_type', 'drmm'),
        nb_supervised_doc=getattr(args, 'nb_supervised_doc', 10),
        doc_topk_term=getattr(args, 'doc_topk_term', 20),
        hist_size=getattr(args, 'hist_size', 30),
        kernel_size=getattr(args, 'kernel_size', 11),
        hidden_size=getattr(args, 'hidden_size', 5)
    )
    
    training_config = TrainingConfig(
        num_epochs=getattr(args, 'num_epochs', 30),
        batch_size=getattr(args, 'batch_size', 20),
        learning_rate=getattr(args, 'learning_rate', 0.001),
        sample_size=getattr(args, 'sample_size', 10),
        num_workers=getattr(args, 'num_workers', 4),
        seed=getattr(args, 'seed', 42)
    )
    
    data_config = DataConfig(
        min_candidates=getattr(args, 'min_candidates', 5),
        require_positive=getattr(args, 'require_positive', True),
        require_negative=getattr(args, 'require_negative', True),
        balance_dataset=getattr(args, 'balance_dataset', False),
        max_queries_per_class=getattr(args, 'max_queries_per_class', None)
    )
    
    inference_config = InferenceConfig(
        score_combination_weight=getattr(args, 'score_combination_weight', 0.7),
        max_workers=getattr(args, 'max_workers', 2),
        run_name=getattr(args, 'run_name', 'nprf')
    )
    
    return NPRFConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        inference=inference_config
    )