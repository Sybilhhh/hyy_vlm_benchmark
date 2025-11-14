from typing import Dict, Any, List, Optional, Tuple
import re
from evaluators.evaluators import BaseEvaluator, EvaluationResult
from tabulate import tabulate


class VideoHallucerEvaluator(BaseEvaluator):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.tasks = ["object_relation", "temporal", "semantic_detail", "interaction", "external_factual", "external_nonfactual", "fact_detect"]
    

    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Main evaluation entry point."""
        
        # Organize predictions by task and source
        organized_preds = self._organize_predictions(predictions)
        
        # Evaluate each task/source combination
        results = {}
        for task in organized_preds:

            samples = organized_preds[task]
            self.logger.info(f"Evaluating {task}: {len(samples)} samples")
            
            try:
                task_results = self._calculate_scores(samples)
                results[task] = task_results    
            except Exception as e:
                self.logger.error(f"Error evaluating {task}: {e}")
                results[task] = {'error': str(e)}
    
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results)
        
        # Generate summary table
        summary_table = self._generate_summary_table(results)
        
        return EvaluationResult(
            score=overall_metrics['overall_score'],
            details={
                'task_results': results,
                'overall_metrics': overall_metrics,
                'summary_table': summary_table
            },
            method='video-hallucer',
            ground_truth_required=True
        )
    
    def _organize_predictions(self, predictions: List[Dict[str, Any]]) -> Dict:
        """Organize predictions by task and source."""
        organized = {}
        
        for pred in predictions:
            task = pred.get('task')
            # source = pred.get('source')
            
            if not task:
                continue
            
            if task not in organized:
                organized[task] = []
            # if source not in organized[task]:
            #     organized[task][source] = [] 
    
            organized[task].append(pred)
        
        return organized
    
    def _calculate_scores(self, samples):
        basic_acc = 0
        halluc_acc = 0
        acc = 0
        for sample in samples:
            basic_hit = 0
            halluc_hit = 0
            final_hit = 0

            basic_answer = sample['reference'][0]
            basic_predict = sample['prediction'][0]
            basic_answer_pattern = r'\b('+basic_answer+ r')\b'
            if re.search(basic_answer_pattern, basic_predict, re.IGNORECASE):
                basic_hit = 1

            halluc_answer = sample['reference'][1]
            halluc_predict = sample['prediction'][1]
            halluc_answer_pattern = r'\b('+halluc_answer+ r')\b'
            if re.search(halluc_answer_pattern, halluc_predict, re.IGNORECASE):
                halluc_hit = 1
            
            final_hit = int(basic_hit and halluc_hit)

            basic_acc += basic_hit
            halluc_acc += halluc_hit
            acc += final_hit
        
        scores = {
            "basic_accuracy": basic_acc / len(samples),
            "halluc_accuracy": halluc_acc / len(samples),
            "accuracy": acc / len(samples),
            "total_samples": len(samples)
        }

        return scores
    
    def _calculate_overall_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate overall metrics across all tasks."""
        metrics = []
        total_basic_acc = 0
        total_halluc_acc = 0
        total_acc = 0
        valid_tasks = 0
        
        # Calculate averages across all tasks
        for task, result in results.items():
            valid_tasks += 1
            total_basic_acc += result.get("basic_accuracy", 0)
            total_halluc_acc += result.get("halluc_accuracy", 0)  
            total_acc += result.get("accuracy", 0)
        
        if valid_tasks > 0:
            avg_basic_acc = total_basic_acc / valid_tasks
            avg_halluc_acc = total_halluc_acc / valid_tasks
            avg_acc = total_acc / valid_tasks
            
            # Overall score is the main accuracy metric
            overall_score = avg_acc
            
            return {
                'overall_score': round(overall_score, 5),
                'basic_accuracy': round(avg_basic_acc, 5),
                'halluc_accuracy': round(avg_halluc_acc, 5), 
                'accuracy': round(avg_acc, 5),
                'fact_score': round((avg_basic_acc + avg_halluc_acc) / 2, 5)
            }
        else:
            return {'overall_score': 0.0}
    
    def _generate_summary_table(self, results: Dict) -> str:
        """Generate summary table for results."""
        if not results:
            return "No results to display"
        
        table_data = [['Task', 'Basic Acc', 'Halluc Acc', 'Final Acc', 'Samples']]
        
        for task, result in results.items():
            if 'error' in result:
                table_data.append([
                    task, 
                    'ERROR', 
                    'ERROR', 
                    'ERROR', 
                    result['error']
                ])
            else:
                table_data.append([
                    task,
                    f"{result.get('basic_accuracy', 0):.3f}",
                    f"{result.get('halluc_accuracy', 0):.3f}", 
                    f"{result.get('accuracy', 0):.3f}",
                    str(result.get('total_samples', 'N/A'))
                ])
        
        # Add overall row if we have valid results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            avg_basic = sum(r['basic_accuracy'] for r in valid_results.values()) / len(valid_results)
            avg_halluc = sum(r['halluc_accuracy'] for r in valid_results.values()) / len(valid_results)
            avg_acc = sum(r['accuracy'] for r in valid_results.values()) / len(valid_results)
            
            table_data.append([
                'OVERALL',
                f"{avg_basic:.3f}",
                f"{avg_halluc:.3f}",
                f"{avg_acc:.3f}",
                str(sum(r.get('total_samples', 0) for r in valid_results.values()))
            ])
        
        return tabulate(table_data, headers='firstrow', tablefmt='grid')