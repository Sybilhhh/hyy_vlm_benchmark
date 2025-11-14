"""
通用的 Pairwise Ranking Evaluator
用于使用 LLM (如 GPT-4o) 对比两个模型的输出质量
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

from evaluators.evaluators import BaseEvaluator, EvaluationResult
from models.models import OpenAIVLMModel


class PairwiseRankingEvaluator(BaseEvaluator):
    """
    通用的成对比较评估器
    
    使用一个强大的 LLM（如 GPT-4o）作为评判，比较两个模型的输出质量
    
    配置参数:
        - model_a_predictions: 模型A的预测结果文件路径
        - model_b_predictions: 模型B的预测结果文件路径
        - model_a_name: 模型A的名称（用于显示）
        - model_b_name: 模型B的名称（用于显示）
        - judge_model: 评判模型名称（默认 'gpt-4o'）
        - evaluation_criteria: 评估标准列表
        - custom_prompt_template: 自定义评估提示模板（可选）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型预测文件路径
        self.model_a_predictions_path = Path(config.get('model_a_predictions'))
        self.model_b_predictions_path = Path(config.get('model_b_predictions'))
        
        # 模型名称
        self.model_a_name = config.get('model_a_name', 'Model A')
        self.model_b_name = config.get('model_b_name', 'Model B')
        
        # 评判模型
        self.judge_model_name = config.get('judge_model', 'gpt-4o')
        
        # 评估标准
        self.evaluation_criteria = config.get('evaluation_criteria', [
            'Accuracy',
            'Detail and Completeness',
            'Technical Terminology',
            'Clarity and Coherence'
        ])
        
        # 自定义提示模板
        self.custom_prompt_template = config.get('custom_prompt_template', None)
        
        # 初始化评判模型客户端
        self.client = OpenAIVLMModel(config)
        
        self.logger.info(f"Initialized PairwiseRankingEvaluator")
        self.logger.info(f"  Model A: {self.model_a_name}")
        self.logger.info(f"  Model B: {self.model_b_name}")
        self.logger.info(f"  Judge: {self.judge_model_name}")
        self.logger.info(f"  Criteria: {', '.join(self.evaluation_criteria)}")
    
    def _build_comparison_prompt(self, prompt: str, response_a: str, response_b: str) -> str:
        """构建比较提示"""
        
        if self.custom_prompt_template:
            # 使用自定义模板
            return self.custom_prompt_template.format(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                model_a_name=self.model_a_name,
                model_b_name=self.model_b_name,
                criteria='\n'.join(f"- {c}" for c in self.evaluation_criteria)
            )
        
        # 默认模板
        criteria_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(self.evaluation_criteria))
        
        comparison_prompt = f"""You are an expert evaluator comparing two AI model responses.

**Original Prompt:**
{prompt}

**{self.model_a_name} Response:**
{response_a}

**{self.model_b_name} Response:**
{response_b}

**Evaluation Criteria:**
{criteria_text}

Please evaluate both responses based on the criteria above and determine which response is better.

**Your response must be in the following JSON format:**
{{
    "winner": "Model A" or "Model B" or "Tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your decision",
    "criteria_scores": {{
        "criterion_name": {{"model_a": score, "model_b": score}},
        ...
    }}
}}

Provide your evaluation:"""
        
        return comparison_prompt
    
    async def _judge_pair(self, sample_id: str, prompt: str, 
                         response_a: str, response_b: str) -> Dict[str, Any]:
        """评判一对响应"""
        
        comparison_prompt = self._build_comparison_prompt(prompt, response_a, response_b)
        
        try:
            # 调用评判模型
            sample = {'prompt': comparison_prompt}
            judgment = await self.client.predict(sample)
            
            # 解析 JSON 响应
            # 尝试提取 JSON（可能被包裹在 markdown 代码块中）
            judgment = judgment.strip()
            if judgment.startswith('```json'):
                judgment = judgment[7:]
            if judgment.startswith('```'):
                judgment = judgment[3:]
            if judgment.endswith('```'):
                judgment = judgment[:-3]
            judgment = judgment.strip()
            
            result = json.loads(judgment)
            
            # 验证必需字段
            if 'winner' not in result or 'confidence' not in result:
                raise ValueError("Missing required fields in judgment")
            
            result['sample_id'] = sample_id
            result['status'] = 'success'
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse judgment for {sample_id}: {e}")
            self.logger.error(f"Raw response: {judgment[:200]}...")
            return {
                'sample_id': sample_id,
                'winner': 'Tie',
                'confidence': 0.0,
                'reasoning': f'Failed to parse judgment: {str(e)}',
                'status': 'parse_error'
            }
        except Exception as e:
            self.logger.error(f"Error judging sample {sample_id}: {e}")
            return {
                'sample_id': sample_id,
                'winner': 'Tie',
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}',
                'status': 'error'
            }
    
    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """
        评估两个模型的预测结果
        
        Note: predictions 参数在这个 evaluator 中不使用，
        因为我们直接从文件加载两个模型的预测
        """
        
        self.logger.info("Loading predictions from files...")
        
        # 加载两个模型的预测
        if not self.model_a_predictions_path.exists():
            raise FileNotFoundError(f"Model A predictions not found: {self.model_a_predictions_path}")
        if not self.model_b_predictions_path.exists():
            raise FileNotFoundError(f"Model B predictions not found: {self.model_b_predictions_path}")
        
        with open(self.model_a_predictions_path, 'r') as f:
            predictions_a = json.load(f)
        
        with open(self.model_b_predictions_path, 'r') as f:
            predictions_b = json.load(f)
        
        self.logger.info(f"Loaded {len(predictions_a)} predictions from {self.model_a_name}")
        self.logger.info(f"Loaded {len(predictions_b)} predictions from {self.model_b_name}")
        
        # 创建 ID 到预测的映射
        preds_a_dict = {p['id']: p for p in predictions_a}
        preds_b_dict = {p['id']: p for p in predictions_b}
        
        # 找到共同的样本
        common_ids = set(preds_a_dict.keys()) & set(preds_b_dict.keys())
        self.logger.info(f"Found {len(common_ids)} common samples to compare")
        
        if len(common_ids) == 0:
            raise ValueError("No common samples found between the two predictions!")
        
        # 进行成对比较
        judgments = []
        for sample_id in sorted(common_ids):
            pred_a = preds_a_dict[sample_id]
            pred_b = preds_b_dict[sample_id]
            
            prompt = pred_a.get('prompt', '')
            response_a = pred_a.get('prediction', '')
            response_b = pred_b.get('prediction', '')
            
            self.logger.info(f"Judging sample: {sample_id}")
            
            judgment = await self._judge_pair(sample_id, prompt, response_a, response_b)
            judgments.append(judgment)
            
            # 添加短暂延迟避免 API 限流
            await asyncio.sleep(0.5)
        
        # 统计结果
        stats = self._compute_statistics(judgments, predictions_a)
        
        # 创建评估结果
        result = EvaluationResult(
            overall_score=stats['win_rate_a'],
            detailed_scores=stats,
            metadata={
                'model_a': self.model_a_name,
                'model_b': self.model_b_name,
                'judge_model': self.judge_model_name,
                'total_comparisons': len(judgments),
                'evaluation_criteria': self.evaluation_criteria
            }
        )
        
        return result
    
    def _compute_statistics(self, judgments: List[Dict[str, Any]], 
                           predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算统计信息"""
        
        total = len(judgments)
        wins_a = sum(1 for j in judgments if j['winner'] == 'Model A')
        wins_b = sum(1 for j in judgments if j['winner'] == 'Model B')
        ties = sum(1 for j in judgments if j['winner'] == 'Tie')
        
        # 按类别统计（如果有 source 字段）
        category_stats = defaultdict(lambda: {'wins_a': 0, 'wins_b': 0, 'ties': 0, 'total': 0})
        
        # 创建 ID 到 source 的映射
        id_to_source = {p['id']: p.get('source', 'unknown') for p in predictions}
        
        for judgment in judgments:
            sample_id = judgment['sample_id']
            category = id_to_source.get(sample_id, 'unknown')
            winner = judgment['winner']
            
            category_stats[category]['total'] += 1
            if winner == 'Model A':
                category_stats[category]['wins_a'] += 1
            elif winner == 'Model B':
                category_stats[category]['wins_b'] += 1
            else:
                category_stats[category]['ties'] += 1
        
        # 计算平均置信度
        avg_confidence = sum(j.get('confidence', 0) for j in judgments) / total if total > 0 else 0
        
        # 构建统计结果
        stats = {
            'total_comparisons': total,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'ties': ties,
            'win_rate_a': wins_a / total if total > 0 else 0,
            'win_rate_b': wins_b / total if total > 0 else 0,
            'tie_rate': ties / total if total > 0 else 0,
            'avg_confidence': avg_confidence,
            'category_breakdown': dict(category_stats),
            'detailed_judgments': judgments
        }
        
        # 生成摘要
        self.logger.info("=" * 60)
        self.logger.info(f"Pairwise Ranking Results: {self.model_a_name} vs {self.model_b_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Comparisons: {total}")
        self.logger.info(f"{self.model_a_name} Wins: {wins_a} ({wins_a/total*100:.1f}%)")
        self.logger.info(f"{self.model_b_name} Wins: {wins_b} ({wins_b/total*100:.1f}%)")
        self.logger.info(f"Ties: {ties} ({ties/total*100:.1f}%)")
        self.logger.info(f"Average Confidence: {avg_confidence:.3f}")
        self.logger.info("=" * 60)
        
        # 按类别统计
        if category_stats:
            self.logger.info("\nBreakdown by Category:")
            for category, cat_stats in sorted(category_stats.items()):
                cat_total = cat_stats['total']
                self.logger.info(f"\n  {category}:")
                self.logger.info(f"    {self.model_a_name}: {cat_stats['wins_a']}/{cat_total} ({cat_stats['wins_a']/cat_total*100:.1f}%)")
                self.logger.info(f"    {self.model_b_name}: {cat_stats['wins_b']}/{cat_total} ({cat_stats['wins_b']/cat_total*100:.1f}%)")
                self.logger.info(f"    Ties: {cat_stats['ties']}/{cat_total} ({cat_stats['ties']/cat_total*100:.1f}%)")
        
        return stats


