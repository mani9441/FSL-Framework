"""
Question Answering Evaluator for FSL Research Framework.
Calculates Correctness (Exact Match, Substring, F1), Completeness, and Relevance.
"""

import re
from typing import List, Dict, Any, Optional, Set
from utilities.logger import get_logger

logger = get_logger("qa_evaluator")


def is_unanswerable_ref(ref: Any) -> bool:
    """Checks if reference indicates an unanswerable question in SQuAD v2."""
    if ref is None:
        return True
    if isinstance(ref, (list, tuple, set)):
        if len(ref) == 0:
            return True
        return all(is_unanswerable_ref(x) for x in ref)
    s = str(ref).strip().lower()
    return s in ["", "[]", "['']", '[""]', "no answer", "unanswerable", "none", "n/a"]


def is_no_answer_pred(pred: str) -> bool:
    """Checks if model prediction indicates that no answer is available in context."""
    if not pred or not str(pred).strip():
        return True
    norm = normalize_answer(pred)
    if norm in [
        "", "none", "no answer", "unanswerable", "na", "no info", "not mentioned",
        "not stated", "cannot find", "not provided", "question is incomplete",
        "not explicitly stated"
    ]:
        return True
    lower = str(pred).lower()
    no_ans_phrases = [
        "no answer", "unanswerable", "not mentioned", "not stated", "cannot find an answer",
        "cannot find the answer", "no information", "is not provided", "does not specify",
        "cannot be answered", "there is no specific mention", "not explicitly stated",
        "none.", "the question is incomplete"
    ]
    return any(phrase in lower for phrase in no_ans_phrases)


def normalize_answer(text: str) -> str:
    """Normalizes answer text for QA evaluation according to benchmark scoring conventions."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove articles (a, an, the)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def compute_token_f1(pred: str, ref: str) -> float:
    """Computes word token overlap F1 score between prediction and reference answer."""
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common_tokens = set(pred_tokens).intersection(set(ref_tokens))
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common_tokens)

    if num_common == 0:
        return 0.0

    precision = num_common / float(len(pred_tokens))
    recall = num_common / float(len(ref_tokens))
    return 2.0 * (precision * recall) / (precision + recall)


class QAEvaluator:
    """Evaluator for Question Answering tasks with SQuAD v2 compatibility."""

    @classmethod
    def evaluate_single(
        cls,
        pred: str,
        ref: Any,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single prediction against ground truth reference(s).
        Returns dictionary containing qa_exact_match, qa_f1, response_correctness,
        response_completeness, response_relevance, and is_correct.
        """
        # 1. Unanswerable SQuAD v2 Reference
        if is_unanswerable_ref(ref):
            if is_no_answer_pred(pred):
                return {
                    "qa_exact_match": 1.0,
                    "qa_f1": 1.0,
                    "response_correctness": 1.0,
                    "response_completeness": 1.0,
                    "response_relevance": 1.0,
                    "is_correct": True,
                }
            else:
                return {
                    "qa_exact_match": 0.0,
                    "qa_f1": 0.0,
                    "response_correctness": 0.0,
                    "response_completeness": 0.0,
                    "response_relevance": 0.0,
                    "is_correct": False,
                }

        # 2. Reference has valid answer
        if is_no_answer_pred(pred):
            return {
                "qa_exact_match": 0.0,
                "qa_f1": 0.0,
                "response_correctness": 0.0,
                "response_completeness": 0.0,
                "response_relevance": 0.0,
                "is_correct": False,
            }

        ref_candidates = [ref] if isinstance(ref, (str, int, float)) else list(ref)
        ref_candidates = [str(r) for r in ref_candidates if str(r).strip()]

        if not ref_candidates:
            # Fallback if empty ref candidates
            return {
                "qa_exact_match": 1.0 if is_no_answer_pred(pred) else 0.0,
                "qa_f1": 1.0 if is_no_answer_pred(pred) else 0.0,
                "response_correctness": 1.0 if is_no_answer_pred(pred) else 0.0,
                "response_completeness": 1.0 if is_no_answer_pred(pred) else 0.0,
                "response_relevance": 1.0 if is_no_answer_pred(pred) else 0.0,
                "is_correct": is_no_answer_pred(pred),
            }

        best_em = 0.0
        best_f1 = 0.0
        best_comp = 0.0
        norm_p = normalize_answer(pred)
        pred_words = set(norm_p.split())

        has_substring = False
        for r_cand in ref_candidates:
            norm_r = normalize_answer(r_cand)
            if norm_p == norm_r:
                best_em = 1.0
            if norm_r and (norm_r in norm_p or norm_p in norm_r):
                has_substring = True

            f1 = compute_token_f1(pred, r_cand)
            if f1 > best_f1:
                best_f1 = f1

            ref_words = set(norm_r.split())
            if ref_words:
                comp = len(ref_words.intersection(pred_words)) / float(len(ref_words))
            else:
                comp = 1.0 if not pred_words else 0.0
            if comp > best_comp:
                best_comp = comp

        # Relevance calculation: context overlap or token F1 fallback
        if context and str(context).strip():
            context_words = set(normalize_answer(context).split())
            if pred_words and context_words:
                rel = len(pred_words.intersection(context_words)) / float(len(pred_words))
            else:
                rel = 0.5
        else:
            rel = best_f1

        is_corr = bool(best_em == 1.0 or best_f1 >= 0.5 or has_substring)

        return {
            "qa_exact_match": round(best_em, 4),
            "qa_f1": round(best_f1, 4),
            "response_correctness": round(best_f1, 4),
            "response_completeness": round(best_comp, 4),
            "response_relevance": round(rel, 4),
            "is_correct": is_corr,
        }

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        references: List[str],
        contexts: Optional[List[str]] = None,
        request_successes: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Computes aggregate QA metrics across prediction/reference lists over valid executions.
        Returns None for metrics if no successful predictions exist.
        """
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("Empty or mismatched predictions/references for QA evaluation.")
            return {
                "qa_exact_match": None,
                "qa_f1": None,
                "exact_match": None,
                "f1": None,
                "response_correctness": None,
                "response_completeness": None,
                "response_relevance": None,
            }

        valid_items = []
        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            succ = request_successes[idx] if request_successes and idx < len(request_successes) else True
            if succ:
                ctx = contexts[idx] if contexts and idx < len(contexts) else None
                valid_items.append((pred, ref, ctx))

        if not valid_items:
            logger.warning("No valid successful execution records for QA evaluation.")
            return {
                "qa_exact_match": None,
                "qa_f1": None,
                "exact_match": None,
                "f1": None,
                "response_correctness": None,
                "response_completeness": None,
                "response_relevance": None,
            }

        em_list = []
        f1_list = []
        comp_list = []
        rel_list = []

        for pred, ref, ctx in valid_items:
            res = cls.evaluate_single(pred, ref, context=ctx)
            em_list.append(res["qa_exact_match"])
            f1_list.append(res["qa_f1"])
            comp_list.append(res["response_completeness"])
            rel_list.append(res["response_relevance"])

        total = float(len(valid_items))
        mean_em = sum(em_list) / total
        mean_f1 = sum(f1_list) / total
        mean_comp = sum(comp_list) / total
        mean_rel = sum(rel_list) / total

        metrics = {
            "qa_exact_match": round(mean_em, 4),
            "qa_f1": round(mean_f1, 4),
            "exact_match": round(mean_em, 4),
            "f1": round(mean_f1, 4),
            "response_correctness": round(mean_f1, 4),
            "response_completeness": round(mean_comp, 4),
            "response_relevance": round(mean_rel, 4),
            "completeness": round(mean_comp, 4),
            "relevance": round(mean_rel, 4),
        }

        logger.info(f"QA Evaluation: Exact Match={mean_em:.4f}, F1={mean_f1:.4f}")
        return metrics
