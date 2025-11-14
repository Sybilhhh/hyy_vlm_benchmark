"""
Evaluator registry and evaluation methods for VLM benchmark system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import asyncio
import statistics
from dataclasses import dataclass

# Import evaluation libraries (may not all be available)


try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    score: float
    details: Dict[str, Any]
    method: str
    ground_truth_required: bool


class BaseEvaluator(ABC):
    """Abstract base class for evaluators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ground_truth_required = config.get('ground_truth_required', True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """
        Evaluate predictions against dataset
        
        Args:
            predictions: List of prediction dictionaries
            dataset: Dataset object containing ground truth (if needed)
            
        Returns:
            EvaluationResult object
        """
        pass
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        return text.lower().strip()



        

# class AutoDQEvaluator(BaseEvaluator):
#     """AutoDQ evaluation method using LLM for quality assessment"""
    
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         if not OPENAI_AVAILABLE:
#             self.logger.warning("OpenAI not available, AutoDQ may not work properly")
        
#         self.llm_model = config.get('llm_model', 'gpt-4o')
#         self.llm_api_key = config.get('llm_api_key')
#         self.client = None
        
#         if not self.llm_api_key:
#             raise ValueError("LLM API key is required for AutoDQ evaluation")
    
#     async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
#         """Evaluate using AutoDQ method"""
#         if not self.client:
#             self.client = test.AsyncOpenAI(api_key=self.llm_api_key)
        
#         scores = []
#         detailed_results = []
        
#         for pred in predictions:
#             if pred.get('prediction') is None:
#                 scores.append(0.0)
#                 detailed_results.append({
#                     'id': pred.get('id'),
#                     'score': 0.0,
#                     'error': pred.get('error', 'No prediction generated')
#                 })
#                 continue
            
#             # Get ground truth
#             ground_truth = pred.get('ground_truth')
#             if not ground_truth:
#                 self.logger.warning(f"No ground truth for sample {pred.get('id')}")
#                 continue
            
#             # Evaluate with LLM
#             score = await self._evaluate_single_with_llm(
#                 pred['prediction'], 
#                 ground_truth,
#                 pred.get('image_path', '')
#             )
            
#             scores.append(score)
#             detailed_results.append({
#                 'id': pred.get('id'),
#                 'score': score,
#                 'prediction': pred['prediction'],
#                 'ground_truth': ground_truth
#             })
        
#         final_score = statistics.mean(scores) if scores else 0.0
        
#         return EvaluationResult(
#             score=final_score,
#             details={
#                 'individual_scores': detailed_results,
#                 'mean_score': final_score,
#                 'std_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
#                 'num_evaluated': len(scores)
#             },
#             method='autodq',
#             ground_truth_required=True
#         )
    
#     async def _evaluate_single_with_llm(self, prediction: str, ground_truth: str, image_path: str = "") -> float:
#         """Evaluate a single prediction using LLM"""
#         try:
#             # AutoDQ prompt template
#             prompt = f"""
# You are an expert evaluator for vision-language model outputs. Please evaluate the quality of a model's prediction against the ground truth caption.

# Ground Truth: "{ground_truth}"
# Model Prediction: "{prediction}"

# Please rate the prediction on a scale of 0.0 to 1.0 based on:
# 1. Semantic accuracy (how well it captures the meaning)
# 2. Content coverage (how much relevant information is included)
# 3. Factual correctness (absence of hallucinations)

# Provide only the numerical score (0.0 to 1.0) without explanation.
# """
            
#             response = await self.client.chat.completions.create(
#                 model=self.llm_model,
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=10,
#                 temperature=0.0
#             )
            
#             score_text = response.choices[0].message.content.strip()
            
#             # Extract numeric score
#             score_match = re.search(r'(\d+\.?\d*)', score_text)
#             if score_match:
#                 score = float(score_match.group(1))
#                 return max(0.0, min(1.0, score))  # Clamp to [0, 1]
#             else:
#                 self.logger.warning(f"Could not parse score from: {score_text}")
#                 return 0.0
                
#         except Exception as e:
#             self.logger.error(f"Error in LLM evaluation: {e}")
#             return 0.0


class DavidsonianSceneGraphEvaluator(BaseEvaluator):
    """Davidsonian Scene Graph evaluation (reference-free)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ground_truth_required = False  # Reference-free method
        
        # This would typically use a specialized model for scene graph extraction
        # For now, we'll implement a simplified version
    
    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Evaluate using Davidsonian Scene Graph approach"""
        scores = []
        detailed_results = []
        
        for pred in predictions:
            if pred.get('prediction') is None:
                scores.append(0.0)
                detailed_results.append({
                    'id': pred.get('id'),
                    'score': 0.0,
                    'error': pred.get('error', 'No prediction generated')
                })
                continue
            
            # Simplified scene graph evaluation
            score = self._evaluate_scene_graph_quality(pred['prediction'])
            
            scores.append(score)
            detailed_results.append({
                'id': pred.get('id'),
                'score': score,
                'prediction': pred['prediction']
            })
        
        final_score = statistics.mean(scores) if scores else 0.0
        
        return EvaluationResult(
            score=final_score,
            details={
                'individual_scores': detailed_results,
                'mean_score': final_score,
                'std_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'num_evaluated': len(scores)
            },
            method='davidsonian_scene_graph',
            ground_truth_required=False
        )
    
    def _evaluate_scene_graph_quality(self, prediction: str) -> float:
        """Simplified scene graph quality assessment"""
        if not prediction:
            return 0.0
        
        # Basic heuristics for scene understanding quality
        score = 0.0
        
        # Length bonus (more detailed descriptions tend to be better)
        word_count = len(prediction.split())
        length_score = min(1.0, word_count / 50)  # Normalize to max 50 words
        score += 0.3 * length_score
        
        # Object detection score (presence of nouns)
        nouns = self._count_objects(prediction)
        object_score = min(1.0, len(nouns) / 10)  # Up to 10 objects
        score += 0.4 * object_score
        
        # Relationship detection score (presence of verbs/prepositions)
        relationships = self._count_relationships(prediction)
        rel_score = min(1.0, len(relationships) / 5)  # Up to 5 relationships
        score += 0.3 * rel_score
        
        return min(1.0, score)
    
    def _count_objects(self, text: str) -> List[str]:
        """Simple object counting based on common object words"""
        object_words = {
            'person', 'people', 'man', 'woman', 'child', 'baby', 'dog', 'cat', 'car', 'house',
            'tree', 'building', 'road', 'sky', 'water', 'table', 'chair', 'book', 'phone',
            'computer', 'ball', 'bird', 'flower', 'grass', 'mountain', 'cloud', 'sun'
        }
        
        words = set(self._preprocess_text(text).split())
        found_objects = words.intersection(object_words)
        return list(found_objects)
    
    def _count_relationships(self, text: str) -> List[str]:
        """Simple relationship counting based on common relationship words"""
        relationship_words = {
            'on', 'in', 'under', 'over', 'next to', 'behind', 'in front of', 'near',
            'holding', 'wearing', 'sitting', 'standing', 'walking', 'running', 'looking'
        }
        
        processed_text = self._preprocess_text(text)
        found_rels = []
        for rel in relationship_words:
            if rel in processed_text:
                found_rels.append(rel)
        
        return found_rels


class BLEUEvaluator(BaseEvaluator):
    """BLEU score evaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not NLTK_AVAILABLE:
            self.logger.warning("NLTK not available, BLEU evaluation may not work")
        
        self.smoothing_function = SmoothingFunction().method4 if NLTK_AVAILABLE else None
    
    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Evaluate using BLEU score"""
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for BLEU evaluation")
        
        scores = []
        detailed_results = []
        
        for pred in predictions:
            if pred.get('prediction') is None or pred.get('ground_truth') is None:
                continue
            
            # Calculate ROUGE scores
            scores = self.scorer.score(pred['ground_truth'], pred['prediction'])
            
            result_item = {
                'id': pred.get('id'),
                'prediction': pred['prediction'],
                'ground_truth': pred['ground_truth']
            }
            
            for rouge_type in self.rouge_types:
                score = scores[rouge_type].fmeasure
                all_scores[rouge_type].append(score)
                result_item[f'{rouge_type}_score'] = score
            
            detailed_results.append(result_item)
        
        # Calculate mean scores
        mean_scores = {}
        for rouge_type in self.rouge_types:
            if all_scores[rouge_type]:
                mean_scores[rouge_type] = statistics.mean(all_scores[rouge_type])
            else:
                mean_scores[rouge_type] = 0.0
        
        # Use ROUGE-L as the main score
        final_score = mean_scores.get('rougeL', mean_scores.get(self.rouge_types[0], 0.0))
        
        return EvaluationResult(
            score=final_score,
            details={
                'individual_scores': detailed_results,
                'mean_scores': mean_scores,
                'rouge_types': self.rouge_types,
                'num_evaluated': len(detailed_results)
            },
            method='rouge',
            ground_truth_required=True
        )


class CIDErEvaluator(BaseEvaluator):
    """CIDEr score evaluation (simplified implementation)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # CIDEr typically requires multiple references, but we'll implement a simplified version
    
    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Evaluate using simplified CIDEr-like scoring"""
        scores = []
        detailed_results = []
        
        # Build document frequency for TF-IDF weighting
        doc_freq = self._build_document_frequency(predictions)
        
        for pred in predictions:
            if pred.get('prediction') is None or pred.get('ground_truth') is None:
                continue
            
            score = self._calculate_cider_score(
                pred['prediction'], 
                pred['ground_truth'], 
                doc_freq
            )
            
            scores.append(score)
            detailed_results.append({
                'id': pred.get('id'),
                'score': score,
                'prediction': pred['prediction'],
                'ground_truth': pred['ground_truth']
            })
        
        final_score = statistics.mean(scores) if scores else 0.0
        
        return EvaluationResult(
            score=final_score,
            details={
                'individual_scores': detailed_results,
                'mean_score': final_score,
                'std_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'num_evaluated': len(scores)
            },
            method='cider',
            ground_truth_required=True
        )
    
    def _build_document_frequency(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Build document frequency for TF-IDF calculation"""
        doc_freq = {}
        
        for pred in predictions:
            if pred.get('ground_truth'):
                words = set(self._preprocess_text(pred['ground_truth']).split())
                for word in words:
                    doc_freq[word] = doc_freq.get(word, 0) + 1
        
        return doc_freq
    
    def _calculate_cider_score(self, prediction: str, ground_truth: str, doc_freq: Dict[str, int]) -> float:
        """Calculate simplified CIDEr score"""
        pred_words = self._preprocess_text(prediction).split()
        gt_words = self._preprocess_text(ground_truth).split()
        
        if not pred_words or not gt_words:
            return 0.0
        
        # Calculate n-gram overlap with TF-IDF weighting (simplified)
        pred_ngrams = self._get_ngrams(pred_words, n=4)
        gt_ngrams = self._get_ngrams(gt_words, n=4)
        
        # Calculate weighted overlap
        overlap_score = 0.0
        total_weight = 0.0
        
        for ngram in set(pred_ngrams + gt_ngrams):
            pred_count = pred_ngrams.count(ngram)
            gt_count = gt_ngrams.count(ngram)
            
            # TF-IDF weight (simplified)
            tf_idf = 1.0 / (1 + doc_freq.get(ngram[0], 1))  # Use first word as proxy
            
            overlap_score += min(pred_count, gt_count) * tf_idf
            total_weight += max(pred_count, gt_count) * tf_idf
        
        return overlap_score / total_weight if total_weight > 0 else 0.0
    
    def _get_ngrams(self, words: List[str], n: int) -> List[tuple]:
        """Get n-grams from word list"""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        return ngrams





# Utility functions
def compare_evaluation_results(results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
    """Compare results from different evaluation methods"""
    comparison = {
        'methods': list(results.keys()),
        'scores': {},
        'ranking': [],
        'ground_truth_methods': [],
        'reference_free_methods': []
    }
    
    for method, result in results.items():
        comparison['scores'][method] = result.score
        
        if result.ground_truth_required:
            comparison['ground_truth_methods'].append(method)
        else:
            comparison['reference_free_methods'].append(method)
    
    # Rank by score
    comparison['ranking'] = sorted(
        comparison['scores'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return comparison


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create evaluator registry
    registry = EvaluatorRegistry()
    
    # List available evaluators
    print("Available evaluators:", registry.list_evaluators())
    
    # Example predictions
    example_predictions = [
        {
            'id': 'test1',
            'prediction': 'A cat sitting on a table',
            'ground_truth': 'A cat is sitting on the wooden table',
            'image_path': 'test1.jpg'
        },
        {
            'id': 'test2', 
            'prediction': 'A person walking in the park',
            'ground_truth': 'A man walking through a beautiful park',
            'image_path': 'test2.jpg'
        }
    ]
    
    async def test_evaluations():
        # Test different evaluators
        evaluators_to_test = ['bleu', 'rouge', 'davidsonian-sg', 'cider']
        
        results = await registry.run_evaluation_suite(
            example_predictions, 
            None,  # No dataset needed for this example
            evaluators_to_test
        )
        
        print("\nEvaluation Results:")
        for method, result in results.items():
            print(f"{method}: {result.score:.4f}")
        
        # Compare results
        comparison = compare_evaluation_results(results)
        print(f"\nRanking: {comparison['ranking']}")
    
        # Run async test
        asyncio.run(test_evaluations())
            
        # Tokenize
        reference = self._preprocess_text(pred['ground_truth']).split()
        candidate = self._preprocess_text(pred['prediction']).split()
        
        if not reference or not candidate:
            score = 0.0
        else:
            score = sentence_bleu(
                [reference], 
                candidate, 
                smoothing_function=self.smoothing_function
            )
        
        scores.append(score)
        detailed_results.append({
            'id': pred.get('id'),
            'score': score,
            'prediction': pred['prediction'],
            'ground_truth': pred['ground_truth']
        })
        
        final_score = statistics.mean(scores) if scores else 0.0
        
        return EvaluationResult(
            score=final_score,
            details={
                'individual_scores': detailed_results,
                'mean_score': final_score,
                'std_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'num_evaluated': len(scores)
            },
            method='bleu',
            ground_truth_required=True
        )


class ROUGEEvaluator(BaseEvaluator):
    """ROUGE score evaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not ROUGE_AVAILABLE:
            self.logger.warning("rouge-score not available, ROUGE evaluation may not work")
        
        self.rouge_types = config.get('rouge_types', ['rouge1', 'rouge2', 'rougeL'])
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True) if ROUGE_AVAILABLE else None
    
    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Evaluate using ROUGE score"""
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score is required for ROUGE evaluation")
        
        all_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        detailed_results = []
        
        for pred in predictions:
            if pred.get('prediction') is None or pred.get('ground_truth') is None:
                continue