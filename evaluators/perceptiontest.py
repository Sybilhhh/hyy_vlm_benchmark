from typing import Dict, Any, List, Optional, Tuple
import re
import string
import random
import copy
import statistics
import torch
import numpy as np
from evaluators.evaluators import BaseEvaluator, EvaluationResult
from nncore.ops import temporal_iou
from tabulate import tabulate


class PerceptionTestEvaluator(BaseEvaluator):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    

    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Main evaluation entry point."""
        
        results = self._calculate_scores(predictions)
    
        return EvaluationResult(
            score=results['accuracy'],
            details={
                'task_results': results,
                'overall_metrics': None,
                'summary_table': None
            },
            method='perception-test',
            ground_truth_required=True
        )
    
    def _calculate_scores(self, samples):
        total_questions = 0
        answered_questions = 0
        correct_questions = 0
        for sample in samples:
            references = sample.get("reference", None)
            predictions = sample.get("prediction", None)
            total_questions += len(references)
            for r, p in zip(references, predictions):
                if p in ['0', '1', '2']:
                    answered_questions += 1
                if r == p:
                    correct_questions += 1
        
        scores = {
            "answered_questions": answered_questions,
            "correct_questions": correct_questions,
            "accuracy": correct_questions / total_questions,
            "total_questions": total_questions,
            "total_videos": len(samples)
        }

        return scores