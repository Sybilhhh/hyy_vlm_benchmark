"""
Test-50 Pairwise Ranking Evaluator
使用 GPT-4o 对比两个模型的输出质量（包含视频内容作为参考）
"""
from typing import Dict, Any, List, Optional
import asyncio
import json
import statistics
import base64
import cv2
import numpy as np
from pathlib import Path
from evaluators.evaluators import BaseEvaluator, EvaluationResult

try:
    from models import OpenAIVLMModel
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class Test50RankingEvaluator(BaseEvaluator):
    """
    对比两个模型的输出，使用 GPT-4o 进行排名评估（包含视频内容作为参考）
    
    特点：
    - GPT-4o 不仅看两个模型的文本描述，还会看实际的视频内容
    - 这样可以评估哪个描述更准确地反映了视频中的摄影和灯光设置
    - 自动从视频中提取关键帧（默认8帧）
    - 如果预测文件中没有 video_path，则退回到纯文本比较
    
    配置示例：
    test-50-ranking:
      type: test-50-ranking
      model_a_predictions: "/path/to/model_a/predictions/predictions.json"
      model_b_predictions: "/path/to/model_b/predictions/predictions.json"
      model_a_name: "qwen2.5vl-7b"
      model_b_name: "qwen2.5vl-32b"
      judge_model: "gpt-4o"
      num_video_frames: 8  # 可选：每个视频采样的帧数（默认8）
      # GPT-4o API配置
      model_name: "gpt-4o"
      endpoint: "..."
      api_version: "..."
      api_key: "..."
    
    注意：预测文件中需要包含 'video_path' 字段，否则将只进行文本比较
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available, Test50RankingEvaluator requires OpenAI")

        # 模型预测文件路径
        self.model_a_predictions = config.get('model_a_predictions')
        self.model_b_predictions = config.get('model_b_predictions')
        
        # 模型名称（用于显示）
        self.model_a_name = config.get('model_a_name', 'Model A')
        self.model_b_name = config.get('model_b_name', 'Model B')
        
        # 评判模型
        self.judge_model = config.get('judge_model', 'gpt-4o')
        
        # 初始化 OpenAI 客户端
        self.client = OpenAIVLMModel(config)
        
        # 超时和重试配置
        self.timeout_s = float(config.get('timeout_s', 120.0))  # 增加到120秒（视频处理需要更多时间）
        self.max_retry = int(config.get('max_retry', 3))
        self.retry_backoff_s = float(config.get('retry_backoff_s', 5.0))
        
        # 视频帧采样配置
        self.num_frames = int(config.get('num_video_frames', 8))  # 采样8帧
        
        # 并发配置
        self.max_concurrent = int(config.get('max_concurrent', 5))  # 最大并发数（默认5）
        self.logger.info(f"Concurrent requests: {self.max_concurrent}")

    async def _ensure_client(self):
        if not await self.client.is_initialized():
            await self.client.load_model()
    
    def _extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """从视频中均匀采样帧"""
        if not Path(video_path).exists():
            self.logger.error(f"Video file not found: {video_path}")
            return []
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            self.logger.error(f"Failed to read video: {video_path}")
            cap.release()
            return []
        
        # 均匀采样
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换为 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        self.logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """将帧编码为 base64 字符串"""
        # 调整大小以节省 token（保持宽高比）
        h, w = frame.shape[:2]
        max_size = 512
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 编码为 JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')

    def _load_predictions(self, prediction_file: str) -> Dict[str, Dict[str, Any]]:
        """加载预测文件并按 ID 索引"""
        path = Path(prediction_file)
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
        
        with open(path, 'r') as f:
            predictions = json.load(f)
        
        # 按 ID 建立索引
        indexed = {}
        for pred in predictions:
            sample_id = pred.get('id')
            if sample_id:
                indexed[sample_id] = pred
        
        self.logger.info(f"Loaded {len(indexed)} predictions from {prediction_file}")
        return indexed

    async def _rank_pair(self, prompt: str, response_a: str, response_b: str, 
                        sample_id: str, video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        使用 GPT-4o 评判两个响应的质量（包含视频内容作为参考）
        
        参数:
        - prompt: 原始任务提示
        - response_a: 模型A的响应
        - response_b: 模型B的响应
        - sample_id: 样本ID
        - video_path: 视频文件路径（可选）
        
        返回:
        - winner: "A", "B", 或 "tie"
        - reason: 评判理由
        - confidence: 置信度 (1-5)
        """
        
        # 如果提供了视频，提取帧
        video_frames = []
        if video_path and Path(video_path).exists():
            video_frames = self._extract_video_frames(video_path)
        
        # 根据是否有视频调整提示
        video_instruction = ""
        if video_frames:
            video_instruction = f"""
**IMPORTANT: You have access to {len(video_frames)} frames from the actual video below (shown after the text). 
Carefully examine the video frames to verify which response more accurately describes:**
- The camera angles, movements, and framing visible in the video
- The lighting setup, techniques, and mood shown in the video
- Any specific visual details mentioned in the responses

Use the video frames as the GROUND TRUTH to judge accuracy.**
"""
        else:
            video_instruction = "**Note: Video frames are not available. Compare based on the quality and plausibility of the descriptions.**"
        
        judge_prompt = f"""You are an expert evaluator comparing two AI model responses for a film production report task.

{video_instruction}

**Task Prompt:**
{prompt}

**Response A ({self.model_a_name}):**
{response_a}

**Response B ({self.model_b_name}):**
{response_b}

**Evaluation Criteria:**
1. **Accuracy**: How well does the response describe the camera and lighting setup visible in the video?
2. **Detail**: Does it provide specific, technical details about camera angles, movements, and lighting techniques?
3. **Completeness**: Does it cover all important aspects visible in the video?
4. **Technical Terminology**: Does it use appropriate film production terminology?
5. **Clarity**: Is the description clear and well-organized?

**Instructions:**
{"Examine the video frames carefully, then " if video_frames else ""}Compare the two responses and determine which one is better for a film production report.

Output your judgment in JSON format:
{{
  "winner": "A" or "B" or "tie",
  "confidence": 1-5 (1=very uncertain, 5=very certain),
  "reason": "Brief explanation of your judgment (2-3 sentences)",
  "scores": {{
    "response_a": {{
      "accuracy": 1-5,
      "detail": 1-5,
      "completeness": 1-5,
      "terminology": 1-5,
      "clarity": 1-5
    }},
    "response_b": {{
      "accuracy": 1-5,
      "detail": 1-5,
      "completeness": 1-5,
      "terminology": 1-5,
      "clarity": 1-5
    }}
  }}
}}

DO NOT PROVIDE ANY OTHER TEXT. Only output the JSON.
"""

        retry = 0
        last_err = None
        await self._ensure_client()

        while retry < self.max_retry:
            try:
                # 构建消息内容
                if video_frames:
                    # 包含视频帧的消息
                    content = [{"type": "text", "text": judge_prompt}]
                    
                    # 添加视频帧（最多8帧以控制token使用）
                    for i, frame in enumerate(video_frames[:8]):
                        frame_b64 = self._encode_frame_to_base64(frame)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_b64}",
                                "detail": "low"  # 使用低分辨率以节省token
                            }
                        })
                    
                    messages = [{"role": "user", "content": content}]
                else:
                    # 仅文本消息（向后兼容）
                    messages = [{"role": "user", "content": judge_prompt}]
                
                resp = await asyncio.wait_for(
                    self.client.get_api.chat.completions.create(
                        model=self.judge_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1024,
                    ),
                    timeout=self.timeout_s
                )
                content = resp.choices[0].message.content or ""
                
                # 解析 JSON
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                result = json.loads(content)
                
                # 验证结果格式
                if 'winner' not in result or result['winner'] not in ['A', 'B', 'tie']:
                    raise ValueError(f"Invalid winner value: {result.get('winner')}")
                
                return {
                    'winner': result['winner'],
                    'confidence': result.get('confidence', 3),
                    'reason': result.get('reason', 'No reason provided'),
                    'scores': result.get('scores', {}),
                    'success': True,
                }
                
            except Exception as e:
                last_err = str(e)
                self.logger.warning(f"Ranking failed for sample {sample_id} (attempt {retry + 1}/{self.max_retry}): {e}")
                retry += 1
                if retry < self.max_retry:
                    await asyncio.sleep(self.retry_backoff_s)
        
        # 如果所有重试都失败，返回错误
        return {
            'winner': 'error',
            'confidence': 0,
            'reason': f'Failed after {self.max_retry} attempts: {last_err}',
            'success': False,
        }

    async def _process_single_sample(self, sample_id: str, pred_a: Dict, pred_b: Dict, 
                                     semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """
        处理单个样本的评判（带并发控制）
        
        参数:
        - sample_id: 样本ID
        - pred_a: 模型A的预测
        - pred_b: 模型B的预测
        - semaphore: 并发控制信号量
        """
        async with semaphore:
            prompt = pred_a.get('prompt', 'No prompt available')
            response_a = pred_a.get('prediction', '')
            response_b = pred_b.get('prediction', '')
            
            # 获取视频路径
            video_path = pred_a.get('video_path') or pred_b.get('video_path')
            
            # 检查是否有内容过滤
            if '[Content filtered' in response_a or '[Content filtered' in response_b:
                self.logger.warning(f"Sample {sample_id} has filtered content, skipping")
                return {
                    'id': sample_id,
                    'source': pred_a.get('source', 'unknown'),
                    'winner': 'skipped',
                    'reason': 'Content filtered by API policy',
                    'success': False,
                }
            
            # 进行排名（包含视频）
            self.logger.info(f"[Concurrent] Ranking sample {sample_id}" + 
                           (f" (with video)" if video_path else " (text only)"))
            
            ranking = await self._rank_pair(prompt, response_a, response_b, sample_id, video_path)
            
            result_record = {
                'id': sample_id,
                'source': pred_a.get('source', 'unknown'),
                'prompt': prompt,
                'response_a': response_a,
                'response_b': response_b,
                'winner': ranking['winner'],
                'confidence': ranking.get('confidence', 0),
                'reason': ranking.get('reason', ''),
                'scores': ranking.get('scores', {}),
                'success': ranking.get('success', False),
            }
            
            self.logger.info(f"[Concurrent] Sample {sample_id}: {ranking['winner']} " +
                           f"(confidence: {ranking.get('confidence', 0)})")
            
            return result_record

    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """
        主评估函数（支持并发处理）
        
        注意：predictions 参数在这里不使用，因为我们直接从配置的文件路径读取
        """
        await self._ensure_client()
        
        # 加载两个模型的预测
        self.logger.info(f"Loading predictions for {self.model_a_name} from {self.model_a_predictions}")
        preds_a = self._load_predictions(self.model_a_predictions)
        
        self.logger.info(f"Loading predictions for {self.model_b_name} from {self.model_b_predictions}")
        preds_b = self._load_predictions(self.model_b_predictions)
        
        # 找到共同的样本 ID
        common_ids = set(preds_a.keys()) & set(preds_b.keys())
        self.logger.info(f"Found {len(common_ids)} common samples to compare")
        
        if len(common_ids) == 0:
            raise ValueError("No common samples found between the two prediction files")
        
        # 创建信号量控制并发数量
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 创建所有任务
        tasks = []
        for sample_id in sorted(common_ids):
            pred_a = preds_a[sample_id]
            pred_b = preds_b[sample_id]
            task = self._process_single_sample(sample_id, pred_a, pred_b, semaphore)
            tasks.append(task)
        
        # 并发执行所有任务
        self.logger.info(f"Starting concurrent evaluation with max {self.max_concurrent} concurrent requests...")
        import time
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Completed {len(results)} comparisons in {elapsed_time:.1f} seconds " +
                        f"({elapsed_time/len(results):.1f}s per comparison)")
        
        # 处理结果和异常
        processed_results = []
        wins_a = 0
        wins_b = 0
        ties = 0
        errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 处理异常
                sample_id = sorted(common_ids)[i]
                self.logger.error(f"Error processing sample {sample_id}: {result}")
                processed_results.append({
                    'id': sample_id,
                    'winner': 'error',
                    'reason': str(result),
                    'success': False,
                })
                errors += 1
            else:
                processed_results.append(result)
                
                # 统计
                winner = result.get('winner')
                if winner == 'A':
                    wins_a += 1
                elif winner == 'B':
                    wins_b += 1
                elif winner == 'tie':
                    ties += 1
                elif winner in ['error', 'skipped']:
                    errors += 1
        
        results = processed_results
        
        # 计算统计信息
        total_valid = wins_a + wins_b + ties
        win_rate_a = wins_a / total_valid if total_valid > 0 else 0
        win_rate_b = wins_b / total_valid if total_valid > 0 else 0
        tie_rate = ties / total_valid if total_valid > 0 else 0
        
        # 按类别统计
        category_stats = self._compute_category_stats(results)
        
        # 生成报告
        summary_table = self._generate_summary_table(
            wins_a, wins_b, ties, errors, total_valid,
            win_rate_a, win_rate_b, tie_rate, category_stats
        )
        
        details = {
            'model_a_name': self.model_a_name,
            'model_b_name': self.model_b_name,
            'total_samples': len(common_ids),
            'valid_comparisons': total_valid,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'ties': ties,
            'errors': errors,
            'win_rate_a': win_rate_a,
            'win_rate_b': win_rate_b,
            'tie_rate': tie_rate,
            'category_stats': category_stats,
            'individual_results': results,
            'summary_table': summary_table,
        }
        
        # 整体评分：使用 Model A 的胜率作为评分（0-1之间）
        overall_score = win_rate_a
        
        return EvaluationResult(
            score=overall_score,
            details=details,
            method='test50_pairwise_ranking',
            ground_truth_required=False,  # 不需要 ground truth，只对比两个模型
        )
    
    def _compute_category_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """按视频类别统计结果"""
        category_data = {}
        
        for result in results:
            if not result.get('success', False):
                continue
            
            source = result.get('source', 'unknown')
            if source not in category_data:
                category_data[source] = {
                    'wins_a': 0,
                    'wins_b': 0,
                    'ties': 0,
                    'total': 0,
                }
            
            category_data[source]['total'] += 1
            winner = result.get('winner')
            if winner == 'A':
                category_data[source]['wins_a'] += 1
            elif winner == 'B':
                category_data[source]['wins_b'] += 1
            elif winner == 'tie':
                category_data[source]['ties'] += 1
        
        # 计算比率
        for cat, stats in category_data.items():
            total = stats['total']
            if total > 0:
                stats['win_rate_a'] = stats['wins_a'] / total
                stats['win_rate_b'] = stats['wins_b'] / total
                stats['tie_rate'] = stats['ties'] / total
        
        return category_data
    
    def _generate_summary_table(
        self, wins_a, wins_b, ties, errors, total_valid,
        win_rate_a, win_rate_b, tie_rate, category_stats
    ) -> str:
        """生成汇总表格"""
        from prettytable import PrettyTable
        
        table = PrettyTable(['Category', f'{self.model_a_name} Wins', f'{self.model_b_name} Wins', 'Ties', 'Total', f'{self.model_a_name} Win Rate'])
        
        # 按类别显示
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            table.add_row([
                cat,
                stats['wins_a'],
                stats['wins_b'],
                stats['ties'],
                stats['total'],
                f"{stats['win_rate_a']:.1%}"
            ])
        
        # 总计
        table.add_row([
            'OVERALL',
            wins_a,
            wins_b,
            ties,
            total_valid,
            f"{win_rate_a:.1%}"
        ])
        
        summary = f"""
===== Test-50 Pairwise Ranking Evaluation =====
Model A: {self.model_a_name}
Model B: {self.model_b_name}
Judge Model: {self.judge_model}

{table}

Summary:
- {self.model_a_name} wins: {wins_a} ({win_rate_a:.1%})
- {self.model_b_name} wins: {wins_b} ({win_rate_b:.1%})
- Ties: {ties} ({tie_rate:.1%})
- Errors: {errors}
- Total valid comparisons: {total_valid}
"""
        print(summary)
        return summary

