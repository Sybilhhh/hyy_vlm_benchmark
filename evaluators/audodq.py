from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import statistics
import re
import json
import ast
from evaluators.evaluators import BaseEvaluator, EvaluationResult

try:
    from models import OpenAIVLMModel
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def _count_f1(recall: float, precision: float) -> float:
    if recall <= 0.0 or precision <= 0.0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # Remove leading ```[lang]? and trailing ```
        t = re.sub(r"^```(?:json|python)?\s*", "", t)
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _best_effort_parse_to_obj(payload: str) -> Any:
    """
    Tries JSON first; falls back to ast.literal_eval for Python-dict string.
    Also fixes a few common issues.
    """
    t = _strip_code_fences(payload)
    # Light cleanups similar to tarsier
    t = re.sub(r'\s+', ' ', t).strip()
    t = t.replace("True", "true").replace("False", "false")
    # Try JSON
    try:
        return json.loads(t)
    except Exception:
        pass
    # Try literal_eval
    try:
        return ast.literal_eval(t)
    except Exception:
        # Last-ditch: try to coerce into a dict if it looks like keyless body
        if not t.startswith("{") and "events" in t:
            t = "{" + t + "}"
            try:
                return json.loads(t)
            except Exception:
                try:
                    return ast.literal_eval(t)
                except Exception:
                    pass
        raise

class AutoDQEvaluator(BaseEvaluator):
    """
    Tarsier/DREAM-style AutoDQ:
      1) Extract up to N atomic motion events from GT and from Prediction via LLM
      2) For GT events: judge relation wrt Prediction (recall side)
      3) For Pred events: judge relation wrt Ground Truth (precision side)
      4) Score = F1(recall, precision), averaged over samples for final metric
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI not available, AutoDQ may not work properly")

        # Models (can be the same)
        self.event_model = config.get('event_model', config.get('llm_model', 'gpt-4o'))
        self.judge_model = config.get('judge_model', config.get('llm_model', 'gpt-4o'))

        self.client = OpenAIVLMModel(config)

        # Behavior
        self.max_events = int(config.get('max_events', 10))
        self.timeout_s = float(config.get('timeout_s', 60.0))
        self.max_retry = int(config.get('max_retry', 6))
        self.retry_backoff_s = float(config.get('retry_backoff_s', 5.0))

        # Positive classes:
        #   Recall uses only 'entailment'
        #   Precision uses 'entailment' and 'neutral' (as in Tarsier code)
        self.pos_classes_recall = config.get('pos_classes_recall', ['entailment'])
        self.pos_classes_precision = config.get('pos_classes_precision', ['entailment', 'neutral'])

        # Optional: use events provided in the sample dict (keys: 'events' or 'gt_events'/'pred_events')
        self.allow_provided_events = bool(config.get('allow_provided_events', True))

    async def _ensure_client(self):
        if not await self.client.is_initialized():
            await self.client.load_model()

    # ---------------- LLM Calls ---------------- #

    async def _chat_completion_with_retry(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        retry = 0
        last_err = None
        await self._ensure_client()

        while retry < self.max_retry:
            try:
                resp = await asyncio.wait_for(
                    self.client.get_api.chat.completions.create(
                        model=model,
                        messages=messages,
                    ),
                    timeout=self.timeout_s
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                last_err = str(e)
                # Retry on common transient errors (similar to Tarsier’s list)
                transient = any(s in last_err for s in [
                    "qpm limit, you can apply for expansion on the platform",
                    "reach token limit, you can apply for expansion on the platform",
                    "Request timed out",
                    "The service is temporarily unable to process your request.",
                    "upstream failed to respond",
                    "502 Bad Gateway",
                    "429 Too Many Requests",
                    "Retrying request to"
                ])
                if not transient:
                    break
                retry += 1
                await asyncio.sleep(self.retry_backoff_s)

        self.logger.error(f"LLM call failed after {self.max_retry} retries: {last_err}")
        raise RuntimeError(last_err or "Unknown LLM error")

    async def _extract_events(self, caption: str) -> List[str]:
        """
        Extract up to self.max_events atomic motion events from caption.
        Output form: {"events": ["...", "...", ...]}
        """
        caption = (caption or "").replace('"', "'")
        prompt = (
                "Bellow is a description of a video clip:\n"
                f"Video Description: {caption}\n\n"

                "Extract at most 10 key events from the above video description paragraph. Requirements\n:"
                "- An event must include an action, motion or movement (NOT STATIC INFORMATION). DON'T repeat same events.\n"
                "- Every event is represented by a brief sentence within 10 words, with a subject, a predicate and optionally an object, avoid unnecessary appearance descriptions.\n"
                "- Every event must be atomic, meaning that it cannot be further split into multiple events.\n"
                "- Scene cuts and camera motions are NOT events.\n"
                "- Substitute pronouns by the nouns they refer to.\n\n"
                "Please generate the response in the form of a Python dictionary string with keys \"events\". The value of \"events\" is a List(str), of which each item is an event. "
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {\"events\": [event1, event2, ...]}"
        )

        raw = await self._chat_completion_with_retry(
            model=self.event_model,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            obj = _best_effort_parse_to_obj(raw)
            events = obj.get("events", [])
            if not isinstance(events, list):
                raise ValueError("Parsed 'events' is not a list")
            # Normalize to strings
            events = [str(e).strip() for e in events if str(e).strip()]
            # Limit to max_events
            return events[: self.max_events]
        except Exception as e:
            self.logger.warning(f"Event extraction parse failed: {e}\nRaw:\n{raw}")
            # As per Tarsier, if extraction fails, treat as empty and let judge fallback to default
            return []

    async def _judge_relationships(self, events: List[str], video_description: str) -> Tuple[int, int, List[Dict[str, str]]]:
        """
        For each event, classify relationship to video_description:
        entailment / neutral / contradiction.
        Returns: (num_positive, num_total, filled_events_list)
        """
        # If no events, Tarsier treats motion_score = 1.0
        if len(events) == 0:
            # Still build a minimal schema for consistency
            filled = [{"event": "", "relationship": "entailment", "reason": "No events provided; treated as trivially matched."}]
            return (1, 1, filled)

        user_prompt = (
            "Given a video description and a list of events. For each event, classify the relationship between the video description and the event into three classes: entailment, neutral, contradiction.\n"
            "- \"entailment\" means that the video description entails the event.\n"
            "- \"contradiction\" means that some detail in the video description contradicts with the event.\n"
            "- \"neutral\" means that the relationship is neither \"entailment\" or \"contradiction\".\n\n"
            f"Video Description:\n{video_description}\n\n"
            f"Events: {events}\n"

            "Output a JSON formed as:\n"
            "{\n"
            "  \"events\": [\n"
            "    {\"event\": \"copy an event here\", \"relationship\": \"put class name here\",  \"reason\": \"give your reason here\"},\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Output:"
        )

        raw = await self._chat_completion_with_retry(
            model=self.judge_model,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Parse robustly and enforce shape
        try:
            obj = _best_effort_parse_to_obj(raw)
            events_filled = obj.get("events", [])
            if not isinstance(events_filled, list) or len(events_filled) == 0:
                raise ValueError("Missing or invalid 'events' in judge output")

            # Tarsier asserts same length OR the (0,1) special case
            if not (len(events_filled) == len(events) or (len(events) == 0 and len(events_filled) == 1)):
                # Try to salvage by truncating/aligning
                events_filled = events_filled[: len(events)]

            # Count positives according to caller’s semantics (set by precision/recall caller)
            # (We don’t decide here; caller will pass which labels are “positive”.)
            return (0, len(events), events_filled)
        except Exception as e:
            self.logger.warning(f"Judge parse failed: {e}\nRaw:\n{raw}")
            # Fall back to neutral for safety to avoid false inflation
            fallback = [{"event": ev, "relationship": "neutral", "reason": "fallback"} for ev in events]
            return (0, len(events), fallback)

    def _count_positives(self, events_filled: List[Dict[str, str]], positive_labels: List[str]) -> int:
        cnt = 0
        for ev in events_filled:
            rel = str(ev.get("relationship", "")).strip().lower()
            if rel in positive_labels:
                cnt += 1
        return cnt

    async def _score_one_sample(
        self,
        pred_caption: str,
        gt_caption: str,
        provided_gt_events: Optional[List[str]] = None,
        provided_pred_events: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns a dict containing:
          score_r, score_p, f1, eval_infos{gt, pred, events_gt, hit_num_recall, events_pred, hit_num_precision}
        """
        gt_text = self._preprocess_text(gt_caption)
        pr_text = self._preprocess_text(pred_caption)
        # 1) Events: use provided if allowed & present, otherwise extract
        if self.allow_provided_events and isinstance(provided_gt_events, list):
            gt_events = [str(e).strip() for e in provided_gt_events if str(e).strip()]
        else:
            gt_events = await self._extract_events(gt_text)
        if self.allow_provided_events and isinstance(provided_pred_events, list):
            pred_events = [str(e).strip() for e in provided_pred_events if str(e).strip()]
        else:
            pred_events = await self._extract_events(pr_text)

        # 2) Recall side: judge GT events against Prediction text
        _, total_r, events_filled_r = await self._judge_relationships(gt_events, pr_text)
        hits_r = self._count_positives(events_filled_r, self.pos_classes_recall)
        score_r = 1.0 if total_r == 0 else hits_r / max(1, total_r)
        
        # 3) Precision side: judge Pred events against Ground Truth text
        _, total_p, events_filled_p = await self._judge_relationships(pred_events, gt_text)
        hits_p = self._count_positives(events_filled_p, self.pos_classes_precision)
        score_p = 1.0 if total_p == 0 else hits_p / max(1, total_p)

        f1 = _count_f1(score_r, score_p)

        if verbose:
            self.logger.info(f"recall: {hits_r}/{total_r} -> {score_r:.3f}; precision: {hits_p}/{total_p} -> {score_p:.3f}; f1={f1:.3f}")

        # eval_infos = {
        #     "gt": gt_text,
        #     "pred": pr_text,
        #     "events_gt": events_filled_r,
        #     "hit_num_recall": f"hit: {hits_r} / {total_r}",
        #     "events_pred": events_filled_p,
        #     "hit_num_precision": f"hit: {hits_p} / {total_p}",
        # }

        return {
            "score_r": score_r,
            "score_p": score_p,
            "f1": f1,
            # "eval_infos": eval_infos,
        }

    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        await self._ensure_client()

        per_sample: List[Dict[str, Any]] = []
        f1_scores: List[float] = []
        failures_by_cat: Dict[str, int] = {}

        for pred in predictions:
            pid = pred.get("id")
            source = pred.get("source", "UNKNOWN")

            pred_text = pred.get("prediction")
            ref_text = pred.get("reference")

            # assert(len(pred_text) == 1)
            # assert(len(ref_text) == 1)

            # pred_text, ref_text = pred_text[0], ref_text[0]

            # Failure: missing required fields
            if not pred_text or not ref_text:
                failures_by_cat[source] = failures_by_cat.get(source, 0) + 1
                per_sample.append({
                    "id": pid,
                    "source": source,
                    "error": "Missing prediction or reference",
                    "success": False,
                })
                continue

            provided_gt_events = pred.get("events") or pred.get("gt_events")
            provided_pred_events = pred.get("pred_events")

            try:
                result = await self._score_one_sample(
                    pred_caption=pred_text,
                    gt_caption=ref_text,
                    provided_gt_events=provided_gt_events,
                    provided_pred_events=provided_pred_events,
                    verbose=bool(self.config.get("verbose", False)),
                )

                sample_rec = {
                    "id": pid,
                    "source": source,
                    "prediction": pred_text,
                    "reference": ref_text,
                    "success": True,
                    # scores only for successes:
                    "score_r": result["score_r"],
                    "score_p": result["score_p"],
                    "f1": result["f1"],
                    # "eval_infos": result["eval_infos"],
                }
                per_sample.append(sample_rec)
                f1_scores.append(result["f1"])

            except Exception as e:
                self.logger.error(f"AutoDQ error for sample {pid}: {e}")
                failures_by_cat[source] = failures_by_cat.get(source, 0) + 1
                per_sample.append({
                    "id": pid,
                    "source": source,
                    "error": str(e),
                    "success": False,
                })

        # Build/print category table (uses successes only)
        category_table_str = self._build_category_table(per_sample, failures_by_cat)

        # Overall = mean F1 of successes only
        final_f1 = statistics.mean(f1_scores) if f1_scores else 0.0

        successes = [x for x in per_sample if x.get("success") is True]
        mean_recall = statistics.mean([x["score_r"] for x in successes]) if successes else 0.0
        mean_precision = statistics.mean([x["score_p"] for x in successes]) if successes else 0.0

        details = {
            "individual_scores": per_sample,                 # includes success flag + eval_infos on success
            "mean_f1": final_f1,
            "std_f1": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
            "num_evaluated": len(successes),
            "category_table": category_table_str,
            "mean_recall": mean_recall,
            "mean_precision": mean_precision,
        }

        return EvaluationResult(
            score=final_f1,
            details=details,
            method='autodq_dream_by_source',
            ground_truth_required=True,
        )
    
    def _build_category_table(
        self,
        per_sample: List[Dict[str, Any]],
        failures_by_cat: Dict[str, int],
    ) -> str:
        """
        Tarsier-style table grouped by `source` with:
        F1, Recall, Precision, Success, Failed.
        Only counts successful samples in metrics.
        """
        from prettytable import PrettyTable

        # successful samples only
        successes = [s for s in per_sample if s.get("success") is True]

        # group successes by category
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for s in successes:
            cat = s.get("source", "UNKNOWN")
            groups.setdefault(cat, []).append(s)

        table = PrettyTable(['Category', 'F1 Score', 'Action Recall', 'Action Precision', 'Success', 'Failed'])

        for cat in sorted(groups.keys()):
            g = groups[cat]
            recalls = [x['score_r'] for x in g]
            precisions = [x['score_p'] for x in g]
            avg_r = statistics.mean(recalls) if recalls else 0.0
            avg_p = statistics.mean(precisions) if precisions else 0.0
            f1 = _count_f1(avg_r, avg_p)
            success = len(g)
            failed = failures_by_cat.get(cat, 0)
            table.add_row([cat, round(f1, 3), round(avg_r, 3), round(avg_p, 3), success, failed])

        # OVERALL across successes only
        all_recalls = [x['score_r'] for x in successes]
        all_precisions = [x['score_p'] for x in successes]
        overall_r = statistics.mean(all_recalls) if all_recalls else 0.0
        overall_p = statistics.mean(all_precisions) if all_precisions else 0.0
        overall_f1 = _count_f1(overall_r, overall_p)
        overall_success = len(successes)
        overall_failed = sum(failures_by_cat.values())

        table.add_row(['OVERALL', round(overall_f1, 3), round(overall_r, 3), round(overall_p, 3),
                    overall_success, overall_failed])

        pretty = "===== AutoDQ Evaluation by Category (source) =====\n" + str(table)
        print(pretty)
        return pretty