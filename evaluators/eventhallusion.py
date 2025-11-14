from typing import Dict, Any, List, Optional, Tuple
import re
import string
import random
import copy
import statistics
import torch
import numpy as np
import openai
from openai import AzureOpenAI
import logging
import time

from evaluators.evaluators import BaseEvaluator, EvaluationResult
from nncore.ops import temporal_iou
from tabulate import tabulate

MISLEADING = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. \
You will receive both the model\'s output and the ground truth event. \
Your task is to determine whether the model\'s description is consistent with the ground truth event. \
If you find any other descriptions unrelated to the ground truth event, answer "no." Otherwise, answer "yes." \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis and reasoning. \
Model output: {}\
Ground-truth event: {}'

ENTIRE = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. \
You will receive both the model\'s output and a ground-truth event. \
Your task is to determine whether the event described in the model\'s output is consistent with the ground-truth event. \
If true, answer "yes." If it is not consistent with the ground-truth event, answer "no." \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis and reasoning. \
Model output: {}\
Ground-truth event: {}'

INTERLEAVE = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. You will receive the output of the tested model and a special event. \
You need to determine whether this special event is mentioned in the output of the model. \
If mentioned, you need to answer "yes", otherwise answer "no". \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis. \
Output: {}\
Unexpected event: {}'


class EventHallusionEvaluator(BaseEvaluator):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.tasks = ["entire", "interleave", "misleading"]
        model_name = config.get("model_name", "")
        endpoint = config.get("endpoint", "")
        api_key = config.get("api_key", "")
        api_version = config.get("api_version", "")
        temparature = config.get("temperature", 1.0)
        top_p = config.get("top_p", 1.0)
        self.gpt_judge = GPTJudge(endpoint, api_key, model_name, api_version)
        self.task_to_prompt = {
            "entire": ENTIRE,
            "interleave": INTERLEAVE,
            "misleading": MISLEADING
        }
    

    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Main evaluation entry point."""
        
        # Organize predictions by task and source
        organized_preds = self._organize_predictions(predictions)
        
        # Evaluate each task/source combination
        results = {}
        for task in organized_preds:
            results[task] = {}

            samples = organized_preds[task]
            self.logger.info(f"Evaluating {task}: {len(samples)} samples")
            
            try:
                task_results_yn = self._evaluate_yes_or_no_questions(samples)
                task_results_desc = self._evaluate_descriptive_questions(samples, self.task_to_prompt[task])
                results[task]["yn"] = task_results_yn
                results[task]["desc"] = task_results_desc    
            except Exception as e:
                self.logger.error(f"Error evaluating {task}: {e}")
                results[task] = {'error': str(e)}
    
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results)
        
        # Generate summary table
        summary_table = self._generate_summary_table(results)
        
        return EvaluationResult(
            score=overall_metrics['overall_score_qa'],
            details={
                'task_results': results,
                'overall_metrics': overall_metrics,
                'summary_table': summary_table
            },
            method='event-hallusion',
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
    
    def _evaluate_yes_or_no_questions(self, samples):
        total_questions = 0
        total_questions_correct = 0
        question_not_match = 0

        for sample in samples:
            reference, answers = sample["reference"][:-1], sample["prediction"][:-1]
            for id_ans, answer in enumerate(answers):
                total_questions += 1

                pred = self._extract_pred(answer)
                if pred is None:
                    question_not_match += 1
                if pred == reference[id_ans]:
                    total_questions_correct += 1

        scores = {
            "accuracy": total_questions_correct / total_questions,
            "not matching rate": question_not_match / total_questions,
            "total questions": total_questions,
            "correct": total_questions_correct
        }

        return scores

    
    def _evaluate_descriptive_questions(self, samples, base_prompt):
        total_description_correct = 0
        gpt_not_match = 0

        for sample in samples:
            reference, answer = sample["reference"][-1], sample["prediction"][-1]

            if len(answer) == 0:
                judgement = ""
            prompt = base_prompt.format(answer, reference)
            judgement = self.gpt_judge.generate_judgement(prompt)
            print(judgement)
            judgement_processed = self._extract_yes_no_gpt(judgement)

            if judgement_processed is None:
                gpt_not_match += 1
            if judgement_processed == "yes":
                total_description_correct += 1

        scores = {
            "accuracy": total_description_correct / len(samples),
            "not matching rate": gpt_not_match / len(samples),
            "total videos": len(samples),
            "correct": total_description_correct
        }

        return scores
    
    
    def _calculate_overall_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate overall metrics across all tasks."""
        total_questions = 0
        total_questions_correct = 0
        total_videos = 0
        total_descriptions_correct = 0
        
        # Calculate averages across all tasks
        for task, result in results.items():
            total_questions += result["yn"].get("total questions", 0)
            total_questions_correct += result["yn"].get("correct", 0)
            total_videos += result["desc"].get("total videos", 0)  
            total_descriptions_correct += result["desc"].get("correct", 0)
            
        return {
            'overall_score_qa': round(total_questions_correct / total_questions, 5),
            'overall_score_description': round(total_descriptions_correct / total_videos, 5)
        }
       
    def _generate_summary_table(self, results: Dict) -> str:
        """Generate summary table for results."""
        if not results:
            return "No results to display"
        
        # Prepare table headers
        headers = [
            "Task", 
            "QA Accuracy", 
            "QA Total", 
            "QA Correct",
            "Desc Accuracy", 
            "Desc Total", 
            "Desc Correct",
            "QA Not Match Rate",
            "Desc Not Match Rate"
        ]
        
        # Prepare table data
        table_data = []
        
        for task_name, task_results in results.items():
            if 'error' in task_results:
                # Handle error case
                row = [
                    task_name,
                    "ERROR",
                    "ERROR", 
                    "ERROR",
                    "ERROR",
                    "ERROR",
                    "ERROR",
                    "ERROR",
                    "ERROR"
                ]
            else:
                # Extract yes/no question results
                yn_results = task_results.get("yn", {})
                qa_accuracy = yn_results.get("accuracy", 0)
                qa_total = yn_results.get("total questions", 0)
                qa_correct = yn_results.get("correct", 0)
                qa_not_match = yn_results.get("not matching rate", 0)
                
                # Extract descriptive question results
                desc_results = task_results.get("desc", {})
                desc_accuracy = desc_results.get("accuracy", 0)
                desc_total = desc_results.get("total videos", 0)
                desc_correct = desc_results.get("correct", 0)
                desc_not_match = desc_results.get("not matching rate", 0)
                
                row = [
                    task_name,
                    f"{qa_accuracy:.4f}",
                    str(qa_total),
                    str(qa_correct),
                    f"{desc_accuracy:.4f}",
                    str(desc_total),
                    str(desc_correct),
                    f"{qa_not_match:.4f}",
                    f"{desc_not_match:.4f}"
                ]
            
            table_data.append(row)
        
        # Add overall metrics row if we have valid results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            overall_metrics = self._calculate_overall_metrics(valid_results)
            overall_row = [
                "OVERALL",
                f"{overall_metrics.get('overall_score_qa', 0):.4f}",
                "-",
                "-", 
                f"{overall_metrics.get('overall_score_description', 0):.4f}",
                "-",
                "-",
                "-",
                "-"
            ]
            table_data.append(overall_row)
        
        # Generate the table using tabulate
        table_str = tabulate(
            table_data, 
            headers=headers, 
            tablefmt="grid",
            stralign="center",
            numalign="center"
        )
        
        return table_str
        
    
    def _extract_pred(self, video_llm_pred):
        if video_llm_pred is None:
            return None
        
        video_llm_pred = video_llm_pred.lower()
        if video_llm_pred.startswith("yes"):
            return "Yes."
        elif video_llm_pred.startswith("no"):
            return "No."
        else:
            return None
        
    def _extract_yes_no_gpt(self, response_text):
        # Remove "**" for bold in gpt's judgement
        response_text = response_text.replace("*", "").lower()
        
        if response_text.startswith("yes"):
            return "yes"
        elif response_text.startswith("no"):
            return "no"
        else:
            return None
        

class GPTJudge():
    def __init__(self, endpoint, api_key, model_name, api_version):
        self.model_name = model_name
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key
        )

    def _get_azure_gpt_response(self, prompt, max_retries=5, retry_delay=5):
        """
        Get response from Azure OpenAI API with retry logic
        """
        data = {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. You are chatting with the user via the ChatGPT iOS app. This means most of the time your lines should be a sentence or two, unless the user's request requires reasoning or long-form outputs. Never use emojis, unless explicitly asked to. Knowledge cutoff: 2023-10 Current date: 2024-08-15. Image input capabilities: Enabled Personality: v2"
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 64,
            "temperature": 1.0,
            "top_p": 1.0,
            "model": self.model_name
        }

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**data)
                return {"choices": [{"message": {"content": response.choices[0].message.content}}]}
            except Exception as e:
                logging.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:  
                    time.sleep(retry_delay)
                else:
                    return {"error": str(e)}

    def generate_judgement(self, prompt):
        response = self._get_azure_gpt_response(prompt)
        if 'error' in response:
            return ""
        else:
            judgement = response.get('choices', [{}])[0].get('message', {}).get('content', 'No judgement available')
            return judgement
        