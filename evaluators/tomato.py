from typing import Dict, Any, List, Optional, Tuple
import re
from evaluators.evaluators import BaseEvaluator, EvaluationResult
from tabulate import tabulate


class TOMATOEvaluator(BaseEvaluator):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.tasks = ["count", "direction", "rotation", "shape&trend", "velocity&frequency", "visual_cues"]
    

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
            method='tomato',
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
        answered_questions = 0
        correct_questions = 0
        for sample in samples:

            options = sample['options']
            answer = sample['reference']
            prediction = sample['prediction']
            if prediction in options:
                answered_questions += 1
            if prediction == options[answer]:
                correct_questions += 1
        
        scores = {
            "answered_questions": answered_questions,
            "correct_questions": correct_questions,
            "accuracy": correct_questions / len(samples),
            "total_questions": len(samples)
        }

        return scores
    
    def _calculate_overall_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate overall metrics across all tasks."""
        total_questions = 0
        total_answered_questions = 0
        total_correct_questions = 0
        
        # Calculate averages across all tasks
        for task, result in results.items():
            total_questions += result.get("total_questions", 0)
            total_answered_questions += result.get("answered_questions", 0)
            total_correct_questions += result.get("correct_questions", 0)
           
        # Overall score is the main accuracy metric
        overall_score = total_correct_questions / total_questions
        
        return {
            'overall_score': round(overall_score, 5),
            'total_questions': total_questions,
            'total_answered_questions': total_answered_questions,
            'total_correct_questions': total_correct_questions,
        }
    
    def _generate_summary_table(self, results: Dict) -> str:
        """Generate summary table for results."""
        if not results:
            return "No results to display"
        
        # Updated table headers to match your actual data structure
        table_data = [['Task', 'Answered', 'Correct', 'Total', 'Accuracy']]
        
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
                    str(result.get('answered_questions', 0)),
                    str(result.get('correct_questions', 0)),
                    str(result.get('total_questions', 0)),
                    f"{result.get('accuracy', 0):.3f}"
                ])
        
        # Add overall row if we have valid results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            total_answered = sum(r.get('answered_questions', 0) for r in valid_results.values())
            total_correct = sum(r.get('correct_questions', 0) for r in valid_results.values())
            total_questions = sum(r.get('total_questions', 0) for r in valid_results.values())
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
            
            table_data.append([
                'OVERALL',
                str(total_answered),
                str(total_correct),
                str(total_questions),
                f"{overall_accuracy:.3f}"
            ])
        
        return tabulate(table_data, headers='firstrow', tablefmt='grid')