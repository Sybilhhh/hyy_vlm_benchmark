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

try:
    import sentence_transformers
    from sentence_transformers.util import dot_score
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class ETBenchEvaluator(BaseEvaluator):
    """
    ET Bench Evaluator for comprehensive video understanding tasks.
    Supports multiple task types:
    - Referring: RAR, ECA, RVQ
    - Grounding: TVG, EPM, TAL, EVS, VHD
    - Captioning: DVC, SLC
    - Complex: TEM, GVQ
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not METRICS_AVAILABLE:
            self.logger.warning("Required metrics libraries not available")
        
        # Initialize sentence transformer for similarity
        self.sentence_transformer = None
        if METRICS_AVAILABLE:
            model_path = config.get('sentence_model', './all-MiniLM-L6-v2')
            self.sentence_transformer = SentenceTransformerSimilarity(model_path)
        
        # Task categories
        self.referring_tasks = ['rar', 'eca', 'rvq']
        self.grounding_tasks = ['tvg', 'epm', 'tal', 'evs', 'vhd']
        self.captioning_tasks = ['dvc', 'slc']
        self.complex_tasks = ['tem', 'gvq']
        
        # IoU thresholds
        self.iou_thresholds = [0.1, 0.3, 0.5, 0.7]
        
        # DVC specific configs
        self.max_proposals = config.get('max_proposals', 1000)
        
        # Whether to use subset evaluation
        self.use_subset = config.get('use_subset', False)
        self.subset_data = None
        if self.use_subset:
            subset_path = config.get('subset_path')
            if subset_path:
                import nncore
                self.subset_data = nncore.load(subset_path)

    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Main evaluation entry point."""
        
        # Organize predictions by task and source
        organized_preds = self._organize_predictions(predictions)
        
        # Evaluate each task/source combination
        results = {}
        for task in organized_preds:
            results[task] = {}
            for source in organized_preds[task]:
                samples = organized_preds[task][source]
                self.logger.info(f"Evaluating {task}_{source}: {len(samples)} samples")
                
                try:
                    if task in self.referring_tasks:
                        task_results = self._evaluate_referring(samples)
                    elif task == 'tvg' or task == 'epm':
                        task_results = self._evaluate_tvg(samples)
                    elif task == 'vhd':
                        task_results = self._evaluate_vhd(samples)
                    elif task == 'tem':
                        task_results = self._evaluate_tem(samples)
                    elif task == 'tal':
                        task_results = self._evaluate_tal(samples)
                    elif task == 'evs':
                        task_results = self._evaluate_evs(samples)
                    elif task in self.captioning_tasks:
                        task_results = self._evaluate_captioning(samples)
                    elif task == 'gvq':
                        task_results = self._evaluate_gvq(samples)
                    else:
                        self.logger.warning(f"Unknown task type: {task}")
                        continue
                    
                    results[task][source] = task_results
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {task}/{source}: {e}")
                    results[task][source] = {'error': str(e)}
        
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
            method='etbench',
            ground_truth_required=True
        )
    
    def _organize_predictions(self, predictions: List[Dict[str, Any]]) -> Dict:
        """Organize predictions by task and source."""
        organized = {}
        
        for pred in predictions:
            task = pred.get('task')
            source = pred.get('source')
            idx = pred.get('idx')
            
            if not task or not source:
                continue
            
            if task not in organized:
                organized[task] = {}
            if source not in organized[task]:
                organized[task][source] = []
            
            # Apply subset filtering if enabled
            if self.use_subset and self.subset_data:
                if source in self.subset_data.get(task, {}) and \
                   idx in self.subset_data[task][source]:
                    organized[task][source].append(pred)
            else:
                organized[task][source].append(pred)
        
        return organized
    
    def _evaluate_referring(self, samples: List[Dict]) -> Dict[str, Any]:
        """Evaluate referring tasks (RAR, ECA, RVQ)."""
        if len(samples[0]['o']) == 4:
            match_map = dict(a=0, b=1, c=2, d=3)
        elif len(samples[0]['o']) == 5:
            match_map = dict(a=0, b=1, c=2, d=3, e=4)
        else:
            raise NotImplementedError("Unsupported number of options")
        
        hit, cnt = 0, 0
        
        for sample in samples:
            gt = sample['p']
            pred = sample['a']
            
            # Try to extract choice from prediction
            ever_matched = False
            match = re.search(r'\(([A-Za-z])\)', pred)
            if match:
                ever_matched = True
                choice = match.group(1).lower()
                if choice in match_map and gt == match_map[choice]:
                    hit += 1
                    continue
            
            # Try alternative parsing
            pred = pred.lower()
            if pred.startswith('best option:'):
                pred = pred[12:]
            
            pred = pred.lstrip().lstrip('(').lstrip()
            if len(pred) == 0:
                cnt += 1
                continue
            
            if len(pred) == 1 or pred[1] in ('.', ',', ' ', ')'):
                ever_matched = True
                if pred[0] in match_map and gt == match_map[pred[0]]:
                    hit += 1
                    continue
            
            # Use similarity matching as fallback
            if self.sentence_transformer:
                hit_idx, max_score = 0, float('-inf')
                _map = ['A', 'B', 'C', 'D', 'E']
                for idx, option in enumerate(sample['o']):
                    if isinstance(option, (list, tuple)):
                        opt = f'{option[0]} - {option[1]}'
                    else:
                        opt = option
                    opt = f'({_map[idx]}) {opt}'
                    sim = self.sentence_transformer.compute_sim(pred, opt)
                    if sim > max_score:
                        hit_idx = idx
                        max_score = sim
                
                if not ever_matched:
                    cnt += 1
                
                if gt == hit_idx:
                    hit += 1
        
        acc = hit / len(samples) if samples else 0
        return {
            'Total': len(samples),
            'Failed': cnt,
            'Acc': round(acc, 5)
        }
    
    def _evaluate_tvg(self, samples: List[Dict]) -> Dict[str, Any]:
        """Evaluate temporal video grounding."""
        hit = [0 for _ in self.iou_thresholds]
        cnt, sum_iou = 0, 0
        
        for sample in samples:
            gt = sample['tgt']
            pred = self._extract_tvg_format(sample['a'])
            
            if pred is None:
                cnt += 1
                continue
            
            pred = pred[0]
            gt = torch.Tensor([gt])
            pred = torch.Tensor([pred])
            iou = temporal_iou(gt, pred).item()
            sum_iou += iou
            
            for i, thr in enumerate(self.iou_thresholds):
                if iou >= thr:
                    hit[i] += 1
        
        recall = [h / len(samples) for h in hit]
        miou = sum_iou / len(samples)
        
        out = {
            'Total': len(samples),
            'Failed': cnt,
            'mIoU': round(miou, 5)
        }
        for rec, thr in zip(recall, self.iou_thresholds):
            out[f'F1@{thr}'] = round(rec, 5)
        out['F1'] = round(sum(recall) / len(recall), 5)
        
        return out
    
    def _evaluate_vhd(self, samples: List[Dict]) -> Dict[str, Any]:
        """Evaluate video highlight detection."""
        hit, cnt = 0, 0
        
        for sample in samples:
            gt = sample['tgt']
            if not isinstance(gt[0][0], (list, tuple)):
                gt = [gt]
            
            match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", sample['a'])
            if not match:
                cnt += 1
                continue
            
            pred = float(match.group(0))
            matched = False
            for annotator in gt:
                for g in annotator:
                    if pred >= g[0] and pred <= g[1]:
                        matched = True
                        break
            if matched:
                hit += 1
        
        return {
            'Total': len(samples),
            'Failed': cnt,
            'F1': round(hit / len(samples), 5)
        }
    
    def _evaluate_tem(self, samples: List[Dict]) -> Dict[str, Any]:
        """Evaluate temporal event matching."""
        hit = [0 for _ in self.iou_thresholds]
        cnt, sum_iou = 0, 0
        
        for sample in samples:
            gt = sample['tgt']
            pred = self._extract_tvg_format(sample['a'])
            
            if pred is None:
                cnt += 1
                continue
            
            pred = pred[0]
            gt = torch.Tensor(gt)
            pred = torch.Tensor([pred])
            iou = temporal_iou(gt, pred).max().item()
            sum_iou += iou
            
            for i, thr in enumerate(self.iou_thresholds):
                if iou >= thr:
                    hit[i] += 1
        
        recall = [h / len(samples) for h in hit]
        miou = sum_iou / len(samples)
        
        out = {
            'Total': len(samples),
            'Failed': cnt,
            'mIoU': round(miou, 5)
        }
        for rec, thr in zip(recall, self.iou_thresholds):
            out[f'R@{thr}'] = round(rec, 5)
        out['mRec'] = round(sum(recall) / len(recall), 5)
        
        return out
    
    def _evaluate_tal(self, samples: List[Dict]) -> Dict[str, Any]:
        """Evaluate temporal action localization."""
        f1_score = [0 for _ in self.iou_thresholds]
        cnt = 0
        
        for sample in samples:
            gt = sample['tgt']
            pred = self._extract_tvg_format(sample['a'])
            
            if pred is None:
                cnt += 1
                continue
            
            gt = torch.Tensor(gt)
            pred = torch.Tensor(pred)
            iou = temporal_iou(gt, pred)
            
            for i, thr in enumerate(self.iou_thresholds):
                if iou.max() < thr:
                    continue
                else:
                    rec = (iou.amax(dim=1) >= thr).float().mean().item()
                    prc = (iou.amax(dim=0) >= thr).float().mean().item()
                    f1_score[i] += 2 * prc * rec / (prc + rec) if (prc + rec) > 0 else 0
        
        f1_score = [f / len(samples) for f in f1_score]
        
        out = {'Total': len(samples), 'Failed': cnt}
        for f1, thr in zip(f1_score, self.iou_thresholds):
            out[f'F1@{thr}'] = round(f1, 5)
        out['F1'] = round(sum(f1_score) / len(f1_score), 5)
        
        return out
    
    def _evaluate_evs(self, samples: List[Dict]) -> Dict[str, Any]:
        """Evaluate event summarization."""
        f1_score = []
        cnt = 0
        
        for sample in samples:
            gt = sample['tgt']
            pred = self._extract_tvg_format(sample['a'])
            
            if pred is None:
                cnt += 1
                continue
            
            gt_map = torch.zeros(1000)
            for g in gt:
                s = max(0, round(g[0]))
                e = round(g[1])
                gt_map[s:e] = 1
            
            pred_map = torch.zeros(1000)
            for p in pred:
                s = max(0, round(p[0]))
                e = round(p[1])
                pred_map[s:e] = 2
            
            com_map = gt_map + pred_map
            
            tp = (com_map == 3).sum().item()
            fp = (com_map == 2).sum().item()
            fn = (com_map == 1).sum().item()
            
            if tp == 0:
                f1 = 0
            else:
                rec = tp / (tp + fn)
                prc = tp / (tp + fp)
                f1 = 2 * prc * rec / (prc + rec)
            
            f1_score.append(f1)
        
        f1_score = round(sum(f1_score) / len(f1_score), 5) if f1_score else 0
        
        return {
            'Total': len(samples),
            'Failed': cnt,
            'F1': f1_score
        }
    
    def _evaluate_captioning(self, samples: List[Dict]) -> Dict[str, Any]:
        """Evaluate dense video captioning."""
        gt_dict, pred = {}, {'results': {}}
        cnt = 0
        
        for sample in samples:
            gt = sample['reference']
            gt_cap = sample['events']
            
            time, cap = self._extract_dvc_format(sample['prediction'])
            if time is None or cap is None:
                cnt += 1
                continue
            
            gt_dict[sample['video_path']] = {
                'timestamps': gt,
                'sentences': gt_cap
            }
            pred['results'][sample['video_path']] = [
                {'sentence': c, 'timestamp': t} for t, c in zip(time, cap)
            ]
        
        scale = len(pred['results']) / len(samples) if samples else 0
        
        if gt_dict and METRICS_AVAILABLE:
            evaluator = DVCEvalWrapper(
                ground_truth=gt_dict,
                prediction=pred,
                tious=self.iou_thresholds,
                sentsim=self.sentence_transformer,
                max_proposals=self.max_proposals
            )
            evaluator.evaluate()
            scores = evaluator.scores
        else:
            scores = {}
            for key in ('Recall', 'Precision', 'Bleu_1', 'Bleu_2', 'Bleu_3', 
                       'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SentSim'):
                scores[key] = [0] * len(self.iou_thresholds)
        
        out = {'Total': len(samples), 'Failed': cnt}
        f1_score = []
        for rec, prc, thr in zip(scores['Recall'], scores['Precision'], 
                                 self.iou_thresholds):
            rec = rec * scale
            prc = prc * scale
            f1 = 0 if prc + rec == 0 else 2 * prc * rec / (prc + rec)
            out[f'F1@{thr}'] = round(f1, 5)
            f1_score.append(f1)
        
        out['F1'] = round(sum(f1_score) / len(f1_score), 5)
        for key in ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 
                   'ROUGE_L', 'CIDEr', 'SentSim'):
            if key in scores:
                out[key] = round(sum(scores[key]) / len(scores[key]), 5)
        
        return out
    
    def _evaluate_gvq(self, samples: List[Dict]) -> Dict[str, Any]:
        """Evaluate grounded video questions."""
        # First evaluate accuracy
        acc_hit_idx = []
        acc_cnt = 0
        
        for sample_idx, sample in enumerate(samples):
            gt = sample['p']
            pred = sample['a']
            
            if pred.lower().startswith('best option:'):
                pred = pred[12:]
            
            pred = pred.lstrip().lstrip('(').lstrip()
            if len(pred) == 0:
                acc_cnt += 1
                continue
            
            if len(sample['o']) == 4:
                match_map = dict(a=0, b=1, c=2, d=3)
            elif len(sample['o']) == 5:
                match_map = dict(a=0, b=1, c=2, d=3, e=4)
            else:
                raise NotImplementedError
            
            if len(pred) == 1 or pred[1] in ('.', ',', ' ', ')'):
                if pred[0].lower() in match_map:
                    if gt == match_map[pred[0].lower()]:
                        acc_hit_idx.append(sample_idx)
                    continue
            
            # Similarity matching
            if self.sentence_transformer:
                hit_idx, max_score = 0, float('-inf')
                _map = ['A', 'B', 'C', 'D', 'E']
                for idx, option in enumerate(sample['o']):
                    if isinstance(option, (list, tuple)):
                        opt = f'{option[0]} - {option[1]}'
                    else:
                        opt = option
                    opt = f'({_map[idx]}) {opt}'
                    sim = self.sentence_transformer.compute_sim(pred, opt)
                    if sim > max_score:
                        hit_idx = idx
                        max_score = sim
                
                if max_score == float('-inf'):
                    acc_cnt += 1
                    continue
                
                if gt == hit_idx:
                    acc_hit_idx.append(sample_idx)
        
        acc_hit_idx = set(acc_hit_idx)
        
        # Evaluate grounding for correct predictions
        hit = [0 for _ in self.iou_thresholds]
        rec_cnt, sum_iou = 0, 0
        
        for sample_idx, sample in enumerate(samples):
            if sample_idx not in acc_hit_idx:
                continue
            
            gt = sample['tgt']
            pred = self._extract_tvg_format(sample['a'])
            
            if pred is None:
                rec_cnt += 1
                continue
            
            pred = pred[0]
            gt = torch.Tensor([gt])
            pred = torch.Tensor([pred])
            iou = temporal_iou(gt, pred).item()
            sum_iou += iou
            
            for i, thr in enumerate(self.iou_thresholds):
                if iou >= thr:
                    hit[i] += 1
        
        recall = [h / len(samples) for h in hit]
        miou = sum_iou / len(samples)
        
        out = {
            'Total': len(samples),
            'Failed': rec_cnt + acc_cnt,
            'mIoU': round(miou, 5)
        }
        for rec, thr in zip(recall, self.iou_thresholds):
            out[f'R@{thr}'] = round(rec, 5)
        out['mRec'] = round(sum(recall) / len(recall), 5)
        out['Acc'] = round(len(acc_hit_idx) / len(samples), 5)
        
        return out
    
    def _extract_tvg_format(self, ans: str) -> Optional[List[List[float]]]:
        """Extract temporal grounding format from answer."""
        ans = ans.lower()
        sentences = re.split(r'[!?\n]', ans)
        
        timestamps = []
        patterns = [r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)"]
        
        for pattern in patterns:
            time_matches = re.findall(pattern, ans)
            if time_matches:
                timestamps = [[float(start), float(end)] for start, end in time_matches]
        
        if len(timestamps) == 0:
            pattern = r"(\d+\.*\d*)\s* to \s*(\d+\.*\d*)"
            time_matches = re.findall(pattern, ans)
            if time_matches:
                timestamps = [[float(start), float(end)] for start, end in time_matches]
        
        # Additional parsing logic for other formats
        if len(timestamps) == 0:
            keywords = ['starts', 'ends', 'happens in', 'start time', 'end time']
            candidates = [s for s in sentences if any(k in s for k in keywords)]
            
            times = []
            time_regex = re.compile(r'\b(\d+\.\d+\b|\b\d+)\b')
            for sentence in candidates:
                time = re.findall(time_regex, sentence)
                if time:
                    times.extend([float(t) for t in time])
            
            times = times[:len(times) // 2 * 2]
            timestamps = [[times[i], times[i + 1]] for i in range(0, len(times), 2)]
        
        results = []
        for (start, end) in timestamps:
            results.append([min(start, end), max(start, end)])
        
        return results if results else None
    
    def _extract_dvc_format(self, caption: str) -> Tuple[Optional[List], Optional[List]]:
        """Extract dense video captioning format."""
        timestamps = []
        sents = []
        
        try:
            timestamps, sents = self._extract_time_from_paragraph(caption)
        except Exception:
            return None, None
        
        if len(timestamps) == 0:
            # Try alternative parsing
            if '\n' in caption:
                caps = [c for c in caption.split('\n') if len(c) > 7]
            else:
                caps = [c + '.' for c in caption.split('.') if len(c) > 7]
            
            for cap in caps:
                try:
                    parts = cap.split('seconds')
                    if len(parts) < 2:
                        continue
                    
                    time_part = parts[0]
                    time_extracted = self._extract_time_part(time_part)
                    if not time_extracted:
                        continue
                    
                    time_part = time_extracted[0]
                    sent_part = parts[-1]
                    
                    stime = round(float(time_part.split('-')[0].strip()), 2)
                    etime = round(float(time_part.split('-')[1].strip()), 2)
                    timestamps.append([stime, etime])
                    sents.append(sent_part.strip())
                except Exception:
                    continue
        
        if len(timestamps) != len(sents) or len(timestamps) == 0:
            return None, None
        
        # Validate timestamps
        for i in range(len(timestamps)):
            timestamps[i] = [min(timestamps[i]), max(timestamps[i])]
        
        return timestamps, sents
    
    def _extract_time_from_paragraph(self, paragraph: str) -> Tuple[List, List]:
        """Extract timestamps and captions from paragraph."""
        paragraph = paragraph.lower()
        patterns = [
            (r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)", r"(\d+\.*\d*\s*-\s*\d+\.*\d*)")
        ]
        timestamps, captions = [], []
        
        for time_pattern, string_pattern in patterns:
            time_matches = re.findall(time_pattern, paragraph)
            string_matches = re.findall(string_pattern, paragraph)
            
            if time_matches:
                timestamps = [[float(start), float(end)] 
                             for start, end in time_matches]
                rest_para = paragraph
                for time_string in string_matches:
                    rest_para = rest_para.replace(time_string, '\n')
                captions = rest_para.replace('seconds', '').split('\n')
            if len(timestamps) > 0:
                break
        
        captions = [c.strip().strip(', ').rstrip() 
                   for c in captions if len(c) > 5]
        min_len = min(len(timestamps), len(captions))
        timestamps = timestamps[:min_len]
        captions = captions[:min_len]
        
        return timestamps, captions
    
    def _extract_time_part(self, time_part: str) -> Optional[List[str]]:
        """Extract time part from string."""
        radius = 20
        extracted = re.compile(r"\d+\.*\d*\s*-\s*\d+\.*\d*").findall(time_part)
        
        if not extracted:
            # Try other formats
            numbers = re.compile(r"\d+\.*\d*(?!\.)").findall(time_part)
            if len(numbers) == 1:
                t = float(numbers[0])
                if t > radius:
                    extracted = [f'{t - radius} - {t + radius}']
                else:
                    extracted = [f'{t} - {t + 2 * radius}']
            elif len(numbers) == 2:
                extracted = [f'{numbers[0]} - {numbers[1]}']
        
        return extracted if extracted else None
    
    def _calculate_overall_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate overall metrics across all tasks."""
        metrics = []
        
        # Referring tasks accuracy
        for task in self.referring_tasks:
            if task in results:
                accs = [r.get('Acc', 0) for r in results[task].values()]
                if accs:
                    metrics.append(('ref_acc', sum(accs) / len(accs)))
        
        # Grounding tasks F1
        for task in self.grounding_tasks:
            if task in results:
                f1s = [r.get('F1', 0) for r in results[task].values()]
                if f1s:
                    metrics.append(('gnd_f1', sum(f1s) / len(f1s)))
        
        # Captioning tasks F1 and similarity
        for task in self.captioning_tasks:
            if task in results:
                f1s = [r.get('F1', 0) for r in results[task].values()]
                sims = [r.get('SentSim', 0) for r in results[task].values()]
                if f1s:
                    metrics.append(('cap_f1', sum(f1s) / len(f1s)))
                if sims:
                    metrics.append(('cap_sim', sum(sims) / len(sims)))
        
        # Complex tasks recall
        for task in self.complex_tasks:
            if task in results:
                recs = [r.get('mRec', 0) for r in results[task].values()]
                if recs:
                    metrics.append(('com_rec', sum(recs) / len(recs)))
        
        overall_score = sum(v for _, v in metrics) / len(metrics) if metrics else 0
        
        return {
            'overall_score': round(overall_score, 5),
            **{k: round(v, 5) for k, v in metrics}
        }
    
    def _generate_summary_table(self, results: Dict) -> str:
        """Generate summary table for results."""
        tables = []
        
        # Referring tasks table
        if any(t in results for t in self.referring_tasks):
            table_data = [('Task', 'Source', 'Total', 'Failed', 'Acc')]
            for task in self.referring_tasks:
                if task in results:
                    for source, res in results[task].items():
                        table_data.append((
                            task, source,
                            res.get('Total', 0),
                            res.get('Failed', 0),
                            res.get('Acc', 0)
                        ))
            tables.append(("Referring Tasks", tabulate(table_data)))
        
        # Similar tables for other task types...
        
        return "\n\n".join([f"{name}:\n{table}" for name, table in tables])


class SentenceTransformerSimilarity:
    """Wrapper for sentence transformer similarity computation."""
    
    def __init__(self, model_path='./all-MiniLM-L6-v2'):
        self.model = sentence_transformers.SentenceTransformer(model_path)
    
    def compute_sim(self, a: str, b: str) -> float:
        """Compute similarity between two sentences."""
        a_emb = self.model.encode([a])
        b_emb = self.model.encode([b])
        score = dot_score(a_emb, b_emb)[0, 0].cpu()
        return float(score)
    
    def compute_score(self, a: Dict, b: Dict) -> Tuple[float, None]:
        """Compute score for dictionary of sentences."""
        assert len(a) == len(b)
        keys = list(a.keys())
        aa, bb = [], []
        for key in keys:
            assert len(a[key]) == len(b[key]) == 1
            aa.append(a[key][0])
            bb.append(b[key][0])
        a_emb = self.model.encode(aa)
        b_emb = self.model.encode(bb)
        score = dot_score(a_emb, b_emb).cpu()
        assert score.shape[0] == score.shape[1]
        score = [score[i, i].item() for i in range(score.shape[0])]
        score = sum(score) / len(score)
        return float(score), None


class DVCEvalWrapper:
    """Wrapper for Dense Video Captioning evaluation."""
    
    def __init__(self, ground_truth, prediction, tious, sentsim, max_proposals=1000):
        self.tious = tious
        self.max_proposals = max_proposals
        self.ground_truths = [ground_truth]
        self.prediction = self._import_prediction(prediction)
        self.ground_truths_keys = list(ground_truth.keys())
        
        # self.tokenizer = PTBTokenizer(verbose=False)
        self.tokenizer = PTBTokenizer()
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Meteor(), 'METEOR'),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
            (sentsim, 'SentSim')
        ]
        self.scores = {}
    
    def _import_prediction(self, prediction):
        results = {}
        for vid_id in prediction['results']:
            results[vid_id] = prediction['results'][vid_id][:self.max_proposals]
        return results
    
    def _iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), 
                   end - start + end_i - start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou
    
    def _get_gt_vid_ids(self):
        vid_ids = set()
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)
    
    def evaluate(self):
        self.scores = {}
        for tiou in self.tious:
            scores = self._evaluate_tiou(tiou)
            for metric, score in scores.items():
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(score)
        
        self.scores['Recall'] = []
        self.scores['Precision'] = []
        for tiou in self.tious:
            precision, recall = self._evaluate_detection(tiou)
            self.scores['Recall'].append(recall)
            self.scores['Precision'].append(precision)
    
    def _evaluate_detection(self, tiou):
        gt_vid_ids = self._get_gt_vid_ids()
        recall = [0] * len(gt_vid_ids)
        precision = [0] * len(gt_vid_ids)
        
        for vid_i, vid_id in enumerate(gt_vid_ids):
            best_recall = 0
            best_precision = 0
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_set_covered = set()
                pred_set_covered = set()
                
                if vid_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[vid_id]):
                        pred_timestamp = pred['timestamp']
                        for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                            if self._iou(pred_timestamp, ref_timestamp) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)
                    
                    new_precision = float(len(pred_set_covered)) / (pred_i + 1) if pred_i >= 0 else 0
                    best_precision = max(best_precision, new_precision)
                
                new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                best_recall = max(best_recall, new_recall)
            
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        
        return sum(precision) / len(precision), sum(recall) / len(recall)
    
    def _evaluate_tiou(self, tiou):
        vid2capid = {}
        res, gts = {}, {}
        cur_res, cur_gts = {}, {}
        unique_index = 0
        gt_vid_ids = self._get_gt_vid_ids()
        
        def remove_nonascii(text):
            return ''.join([i if ord(i) < 128 else ' ' for i in text])
        
        def random_string(length):
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for _ in range(length))
        
        for vid_id in gt_vid_ids:
            vid2capid[vid_id] = []
            
            if vid_id not in self.prediction:
                pass
            else:
                for pred in self.prediction[vid_id]:
                    has_added = False
                    for gt in self.ground_truths:
                        if vid_id not in gt:
                            continue
                        gt_captions = gt[vid_id]
                        for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):
                            if self._iou(pred['timestamp'], caption_timestamp) >= tiou:
                                cur_res[unique_index] = [{
                                    'caption': remove_nonascii(pred['sentence'])
                                }]
                                cur_gts[unique_index] = [{
                                    'caption': remove_nonascii(gt_captions['sentences'][caption_idx])
                                }]
                                vid2capid[vid_id].append(unique_index)
                                unique_index += 1
                                has_added = True
                    
                    if not has_added:
                        cur_res[unique_index] = [{
                            'caption': remove_nonascii(pred['sentence'])
                        }]
                        cur_gts[unique_index] = [{
                            'caption': random_string(random.randint(10, 20))
                        }]
                        vid2capid[vid_id].append(unique_index)
                        unique_index += 1
        
        output = {}
        for scorer, method in self.scorers:
            all_scores = {}
            
            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)
            
            for vid in vid2capid.keys():
                res[vid] = {index: tokenize_res[index] for index in vid2capid[vid]}
                gts[vid] = {index: tokenize_gts[index] for index in vid2capid[vid]}
            
            for vid_id in gt_vid_ids:
                if len(res.get(vid_id, {})) == 0 or len(gts.get(vid_id, {})) == 0:
                    if isinstance(method, list):
                        score = [0] * len(method)
                    else:
                        score = 0
                else:
                    if isinstance(method, list):
                        score, scores = scorer.compute_score(gts[vid_id], res[vid_id], verbose=0)
                    else:
                        score, scores = scorer.compute_score(gts[vid_id], res[vid_id])
                
                all_scores[vid_id] = score
            
            if isinstance(method, list):
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    output[method[m]] = scores[m]
            else:
                output[method] = np.mean(list(all_scores.values()))
        
        return output