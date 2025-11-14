from .evaluators import *
from .audodq import *
from .etbench import *
from .videohallucer import *
from .eventhallusion import *
from .etva import *
from .perceptiontest import *
# from .tvbench import *
from .tomato import *
from .test50_ranking import *
from .test50_dense_ranking import *
from .pairwise_ranking import *

class EvaluatorRegistry:
    """Registry for managing different evaluation methods"""
    
    def __init__(self):
        self.evaluators: Dict[str, BaseEvaluator] = {}
        self.evaluator_classes = {
            'autodq': AutoDQEvaluator,
            'davidsonian_scene_graph': DavidsonianSceneGraphEvaluator,
            'davidsonian-sg': DavidsonianSceneGraphEvaluator,  # Alias
            'bleu': BLEUEvaluator,
            'rouge': ROUGEEvaluator,
            'cider': CIDErEvaluator,
            'etbench': ETBenchEvaluator,
            'video-hallucer': VideoHallucerEvaluator,
            'event-hallusion': EventHallusionEvaluator,
            'perception-test': PerceptionTestEvaluator,
            # 'tvbench': TVBenchEvaluator,
            'tomato': TOMATOEvaluator,
            'etva': ETVAEvaluator,
            'test-50-ranking': Test50RankingEvaluator,
            'test50-ranking': Test50RankingEvaluator,  # Alias
            'test-50-dense-ranking': Test50DenseRankingEvaluator,
            'test50-dense-ranking': Test50DenseRankingEvaluator,  # Alias
            'pairwise-ranking': PairwiseRankingEvaluator,
            'pairwise': PairwiseRankingEvaluator,  # Alias
        }
        self.logger = logging.getLogger(__name__)
    
    def register_evaluator(self, name: str, evaluator: BaseEvaluator):
        """Register an evaluator instance"""
        self.evaluators[name] = evaluator
        self.logger.info(f"Evaluator '{name}' registered")
    
    def list_evaluators(self) -> List[str]:
        """List all available evaluators"""
        return list(set(list(self.evaluators.keys()) + list(self.evaluator_classes.keys())))
    
    def load_evaluator(self, name: str, config: Dict[str, Any]) -> BaseEvaluator:
        """Load an evaluator by name"""
        if name in self.evaluators:
            return self.evaluators[name]
        
        # Normalize name
        normalized_name = name.lower().replace('-', '_')
        
        # Find evaluator class
        if normalized_name in self.evaluator_classes:
            evaluator_class = self.evaluator_classes[normalized_name]
        elif name in self.evaluator_classes:
            evaluator_class = self.evaluator_classes[name]
        else:
            raise ValueError(f"Unknown evaluator: {name}")
        
        # Create evaluator instance
        evaluator = evaluator_class(config)
        
        # Register and return
        self.register_evaluator(name, evaluator)
        return evaluator
    
    def register_evaluator_class(self, name: str, evaluator_class):
        """Register a new evaluator class"""
        self.evaluator_classes[name] = evaluator_class
        self.logger.info(f"Evaluator class '{name}' registered")
    
    def get_evaluator_info(self, name: str) -> Dict[str, Any]:
        """Get information about an evaluator"""
        if name in self.evaluators:
            evaluator = self.evaluators[name]
            return {
                'name': name,
                'ground_truth_required': evaluator.ground_truth_required,
                'class': evaluator.__class__.__name__,
                'config': evaluator.config
            }
        elif name in self.evaluator_classes:
            return {
                'name': name,
                'class': self.evaluator_classes[name].__name__,
                'available': True
            }
        else:
            return {'name': name, 'available': False}
    
    async def run_evaluation_suite(
        self, 
        predictions: List[Dict[str, Any]], 
        dataset,
        evaluator_names: List[str]
    ) -> Dict[str, EvaluationResult]:
        """Run multiple evaluations"""
        results = {}
        
        for name in evaluator_names:
            try:
                self.logger.info(f"Running evaluation: {name}")
                evaluator = self.load_evaluator(name, {})
                result = await evaluator.evaluate(predictions, dataset)
                results[name] = result
                self.logger.info(f"Completed evaluation {name}: score={result.score:.4f}")
            except Exception as e:
                self.logger.error(f"Error in evaluation {name}: {e}")
                results[name] = EvaluationResult(
                    score=0.0,
                    details={'error': str(e)},
                    method=name,
                    ground_truth_required=True
                )
        
        return results