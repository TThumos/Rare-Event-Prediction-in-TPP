import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualDataProcessor:
    """Processor for creating residual datasets based on weight thresholds"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Value cutoff for keeping residual elements (default: 0.5)
        """
        self.threshold = threshold
        self._validate_threshold()

    
    def _validate_threshold(self):
        """Ensure threshold is in valid range"""
        if not 0 < self.threshold < 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")

    
    def calculate_residual_indices(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Calculate indices to keep based on weight thresholds
        
        Args:
            weights: List of weight arrays for a dataset
            
        Returns:
            List of index arrays specifying elements to retain
        """
        indices = []
        for weight_array in weights:
            if len(weight_array) < 2:
                indices.append(np.array([0]))
                continue

            # Always keep first and last elements
            keep_indices = [0, len(weight_array)-1]
            
            # Find elements below threshold between first and last
            middle_indices = np.where(weight_array[1:-1] < self.threshold)[0] + 1
            indices.append(np.unique(np.concatenate([keep_indices, middle_indices])))
            
        return indices
    
    
    def calculate_kept_indices(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Calculate indices to keep based on weight thresholds
        
        Args:
            weights: List of weight arrays for a dataset
            
        Returns:
            List of index arrays specifying elements to retain
        """
        indices = []
        for weight_array in weights:
            
            # Find elements below threshold between first and last
            middle_indices = np.where(weight_array[1:-1] >= self.threshold)[0] + 1
            indices.append(np.unique(middle_indices))
            
        return indices

    
    def filter_dataset(self, 
                      raw_data: Dict[str, Any], 
                      indices: List[np.ndarray]) -> Dict[str, List]:
        """
        Filter dataset sequences using calculated indices
        
        Args:
            raw_data: Original dataset dictionary
            indices: List of index arrays to apply
            
        Returns:
            Filtered dataset dictionary
        """
        return {
            'time_seqs': [
                [raw_data['time_seqs'][i][j] for j in idx] 
                for i, idx in enumerate(indices)
            ],
            'type_seqs': [
                [raw_data['type_seqs'][i][j] for j in idx] 
                for i, idx in enumerate(indices)
            ],
            'time_delta_seqs': [
                [raw_data['time_delta_seqs'][i][j] for j in idx] 
                for i, idx in enumerate(indices)
            ]
        }

    
    def convert_to_serializable_format(self, 
                                      filtered_data: Dict[str, List], 
                                      dim_process: int,
                                      split) -> Dict[str, Any]:
        """
        Convert filtered data to final serialization format
        
        Args:
            filtered_data: Filtered dataset from filter_dataset()
            dim_process: Number of event types
            
        Returns:
            Dictionary ready for pickle serialization
        """
        return {
            'dim_process': dim_process,
            split: [
                [
                    {
                        'type_event': event_type,
                        'time_since_start': time,
                        'time_since_last_event': delta
                    }
                    for idx, (event_type, time, delta) in enumerate(zip(
                        filtered_data['type_seqs'][i],
                        filtered_data['time_seqs'][i],
                        filtered_data['time_delta_seqs'][i]
                    ))
                ]
                for i in range(len(filtered_data['time_seqs']))
            ]
        }

    
    def process_and_save(self,
                        weights: Dict[str, List[np.ndarray]],
                        raw_datasets: Dict[str, Dict],
                        output_dir_residual: Path,
                        output_dir_kept: Path,
                        dim_process: int):
        """
        Full processing pipeline with file saving
        
        Args:
            weights: Dictionary containing train/valid/test weights
            raw_datasets: Original datasets dictionary
            output_dir: Path for output files
            dim_process: Number of event types in the data
        """
        output_dir_residual.mkdir(parents=True, exist_ok=True)
        output_dir_kept.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'valid', 'test']:
            logger.info(f"Processing {split} dataset...")
            
            # Calculate residual indices
            residual_indices = self.calculate_residual_indices(weights[split])
            kept_indices = self.calculate_kept_indices(weights[split])
            
            # Filter dataset
            filtered_residual = self.filter_dataset(raw_datasets[split], residual_indices)
            filtered_kept = self.filter_dataset(raw_datasets[split], kept_indices)

            # Convert format
            if split == "valid":
                split = "dev"
            processed_residual = self.convert_to_serializable_format(filtered_residual, dim_process, split)
            processed_kept = self.convert_to_serializable_format(filtered_kept, dim_process, split)

            # Save results
            output_path_residual = output_dir_residual / f"{split}.pkl"
            with output_path_residual.open('wb') as f:
                pickle.dump(processed_residual, f)
            logger.info(f"Saved {split} residual data to {output_path_residual}")

            output_path_kept = output_dir_kept / f"{split}.pkl"
            with output_path_kept.open('wb') as f:
                pickle.dump(processed_kept, f)
            logger.info(f"Saved {split} kept data to {output_path_kept}")