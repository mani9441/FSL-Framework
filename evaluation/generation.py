"""
Text Generation Evaluator for FSL Research Framework.
Calculates BLEU (BLEU-1, BLEU-2, BLEU-4) and ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) metrics.
"""

from typing import Optional
import math
import re
from typing import List, Dict, Any, Tuple
from utilities.logger import get_logger

logger = get_logger("generation_evaluator")


def tokenize(text: str) -> List[str]:
    """Tokenizes text into words."""
    if not text:
        return []
    clean = re.sub(r"[^\w\s]", "", text.lower())
    return clean.split()


def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    """Extracts n-grams and frequencies."""
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def compute_bleu_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    """Computes modified n-gram precision."""
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    pred_grams = get_ngrams(pred_tokens, n)
    ref_grams = get_ngrams(ref_tokens, n)

    clipped_count = 0
    for gram, count in pred_grams.items():
        clipped_count += min(count, ref_grams.get(gram, 0))

    total_pred_grams = sum(pred_grams.values())
    return clipped_count / float(total_pred_grams) if total_pred_grams > 0 else 0.0


def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Computes Longest Common Subsequence length between two token lists."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


class GenerationEvaluator:
    """Evaluator for Text Generation and Summarization tasks."""

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        references: List[str],
        request_successes: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Computes BLEU-1, BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, and ROUGE-L over valid predictions.
        Returns None for metrics if no successful predictions exist.
        """
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("Empty or mismatched predictions/references for generation evaluation.")
            return {
                "bleu": None,
                "bleu_1": None,
                "bleu_2": None,
                "bleu_4": None,
                "rouge": None,
                "rouge_1": None,
                "rouge_2": None,
                "rouge_l": None,
            }

        # Filter out failed request executions if specified
        valid_pairs = []
        if request_successes is not None:
            for pred, ref, succ in zip(predictions, references, request_successes):
                if succ:
                    valid_pairs.append((pred, ref))
        else:
            valid_pairs = list(zip(predictions, references))

        if not valid_pairs:
            logger.warning("No valid successful execution records for generation evaluation.")
            return {
                "bleu": None,
                "bleu_1": None,
                "bleu_2": None,
                "bleu_4": None,
                "rouge": None,
                "rouge_1": None,
                "rouge_2": None,
                "rouge_l": None,
            }

        bleu_1_scores, bleu_2_scores, bleu_4_scores = [], [], []
        rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []

        for pred, ref in valid_pairs:
            p_toks = tokenize(pred)
            r_toks = tokenize(ref)

            # BLEU n-gram precision
            b1 = compute_bleu_n(p_toks, r_toks, 1)
            b2 = compute_bleu_n(p_toks, r_toks, 2)
            b4 = compute_bleu_n(p_toks, r_toks, 4)

            # Brevity penalty
            c, r = len(p_toks), len(r_toks)
            bp = math.exp(1 - r / float(c)) if c > 0 and c < r else 1.0

            bleu_1_scores.append(bp * b1)
            bleu_2_scores.append(bp * b2)
            bleu_4_scores.append(bp * b4)

            # ROUGE-1 (Unigram F1)
            r1_prec = compute_bleu_n(p_toks, r_toks, 1)
            r1_rec = compute_bleu_n(r_toks, p_toks, 1)
            rouge_1 = (2 * r1_prec * r1_rec / (r1_prec + r1_rec)) if (r1_prec + r1_rec) > 0 else 0.0
            rouge_1_scores.append(rouge_1)

            # ROUGE-2 (Bigram F1)
            r2_prec = compute_bleu_n(p_toks, r_toks, 2)
            r2_rec = compute_bleu_n(r_toks, p_toks, 2)
            rouge_2 = (2 * r2_prec * r2_rec / (r2_prec + r2_rec)) if (r2_prec + r2_rec) > 0 else 0.0
            rouge_2_scores.append(rouge_2)

            # ROUGE-L (LCS F1)
            lcs = compute_lcs_length(p_toks, r_toks)
            rl_prec = lcs / float(c) if c > 0 else 0.0
            rl_rec = lcs / float(r) if r > 0 else 0.0
            rouge_l = (2 * rl_prec * rl_rec / (rl_prec + rl_rec)) if (rl_prec + rl_rec) > 0 else 0.0
            rouge_l_scores.append(rouge_l)

        total = float(len(valid_pairs))
        m_b1 = sum(bleu_1_scores) / total
        m_b2 = sum(bleu_2_scores) / total
        m_b4 = sum(bleu_4_scores) / total

        m_r1 = sum(rouge_1_scores) / total
        m_r2 = sum(rouge_2_scores) / total
        m_rl = sum(rouge_l_scores) / total

        metrics = {
            "bleu": round(m_b4, 4),
            "bleu_1": round(m_b1, 4),
            "bleu_2": round(m_b2, 4),
            "bleu_4": round(m_b4, 4),
            "rouge": round(m_r1, 4),
            "rouge_1": round(m_r1, 4),
            "rouge_2": round(m_r2, 4),
            "rouge_l": round(m_rl, 4),
        }

        logger.info(f"Generation Evaluation: BLEU-4={m_b4:.4f}, ROUGE-1={m_r1:.4f}, ROUGE-L={m_rl:.4f}")
        return metrics
