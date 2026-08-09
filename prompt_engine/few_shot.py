"""
Few-Shot Context Builder for FSL Research Framework.
Handles candidate selection (1-shot, 3-shot, 5-shot), diversity strategies, and ordering strategies.
"""

import re
import random
from typing import List, Dict, Any, Set
from datasets.schema import StandardRecord
from prompt_engine.templates import format_single_demonstration
from utilities.logger import get_logger

logger = get_logger("few_shot_builder")


def compute_word_jaccard_similarity(text1: str, text2: str) -> float:
    """Computes Jaccard similarity between word sets of two texts, ignoring punctuation."""
    t1_clean = re.sub(r"[^\w\s]", "", text1.lower())
    t2_clean = re.sub(r"[^\w\s]", "", text2.lower())
    words1: Set[str] = set(t1_clean.split())
    words2: Set[str] = set(t2_clean.split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / float(len(union))


class ExampleSelector:
    """Selects and orders few-shot demonstration candidates."""

    @classmethod
    def select_examples(
        cls,
        candidate_pool: List[StandardRecord],
        query_record: StandardRecord,
        num_examples: int = 3,
        diversity: str = "medium",
        ordering: str = "random",
        seed: int = 42,
    ) -> List[StandardRecord]:
        """
        Selects `num_examples` candidates from `candidate_pool` using diversity and ordering strategies.
        """
        if not candidate_pool or num_examples <= 0:
            return []

        if len(candidate_pool) <= num_examples:
            selected = list(candidate_pool)
        else:
            selected = cls._apply_diversity_selection(
                candidate_pool, query_record, num_examples, diversity, seed
            )

        # Apply ordering strategy
        ordered = cls._apply_ordering_strategy(selected, query_record, ordering, seed)
        logger.info(
            f"Selected {len(ordered)} demonstration examples (shot_count={num_examples}, diversity='{diversity}', ordering='{ordering}')"
        )
        return ordered

    @classmethod
    def _apply_diversity_selection(
        cls,
        candidate_pool: List[StandardRecord],
        query_record: StandardRecord,
        num_examples: int,
        diversity: str,
        seed: int,
    ) -> List[StandardRecord]:
        """Filters candidate pool based on diversity level."""
        rng = random.Random(seed)

        if diversity == "low":
            # Select examples with highest similarity to target query
            scored = [
                (compute_word_jaccard_similarity(query_record.input_text, rec.input_text), rec)
                for rec in candidate_pool
            ]
            # Sort descending by similarity, breaking ties with rec.id
            scored.sort(key=lambda x: (x[0], x[1].id), reverse=True)
            return [rec for _, rec in scored[:num_examples]]

        elif diversity == "high":
            # Select maximally diverse examples across classes or distinct texts
            if query_record.task_type == "text_classification":
                label_groups: Dict[str, List[StandardRecord]] = {}
                for rec in candidate_pool:
                    label_groups.setdefault(rec.label, []).append(rec)

                selected: List[StandardRecord] = []
                labels = sorted(list(label_groups.keys()))
                idx = 0
                while len(selected) < num_examples and label_groups:
                    target_label = labels[idx % len(labels)]
                    if label_groups[target_label]:
                        selected.append(label_groups[target_label].pop(0))
                    idx += 1
                return selected[:num_examples]

            else:
                # Textually diverse selection: greedy min similarity
                selected = [candidate_pool[0]]
                remaining = list(candidate_pool[1:])
                while len(selected) < num_examples and remaining:
                    best_candidate = None
                    min_sim = 1.0
                    for cand in remaining:
                        sims = [compute_word_jaccard_similarity(cand.input_text, s.input_text) for s in selected]
                        avg_sim = sum(sims) / len(sims)
                        if avg_sim <= min_sim:
                            min_sim = avg_sim
                            best_candidate = cand
                    if best_candidate:
                        selected.append(best_candidate)
                        remaining.remove(best_candidate)
                    else:
                        break
                return selected[:num_examples]

        else:  # "medium" - balanced candidate selection preserving pool ordering
            step = max(1, len(candidate_pool) // num_examples)
            selected = candidate_pool[::step][:num_examples]
            return selected

    @classmethod
    def _apply_ordering_strategy(
        cls,
        examples: List[StandardRecord],
        query_record: StandardRecord,
        ordering: str,
        seed: int,
    ) -> List[StandardRecord]:
        """Orders selected demonstration examples."""
        if ordering == "original":
            return list(examples)

        elif ordering == "random":
            rng = random.Random(seed)
            shuffled = list(examples)
            rng.shuffle(shuffled)
            return shuffled

        elif ordering in ["similarity_ascending", "performance_based"]:
            if ordering == "performance_based":
                logger.debug("Ordering strategy 'performance_based' executes lexical similarity ascending ordering.")
            # Sort by similarity score ascending so most relevant example is last (closest to query)
            scored = [
                (compute_word_jaccard_similarity(query_record.input_text, rec.input_text), rec)
                for rec in examples
            ]
            scored.sort(key=lambda x: (x[0], x[1].id))  # Ascending similarity order with tie breaker
            return [rec for _, rec in scored]

        else:
            logger.warning(f"Unknown ordering strategy '{ordering}'. Defaulting to original order.")
            return list(examples)


class ContextBuilder:
    """Engine merging selected demonstration examples into context blocks."""

    @classmethod
    def build_few_shot_context(
        cls,
        candidate_pool: List[StandardRecord],
        query_record: StandardRecord,
        num_examples: int = 3,
        diversity: str = "medium",
        ordering: str = "random",
        seed: int = 42,
    ) -> List[StandardRecord]:
        """
        Selects, orders, and returns demonstration records ready for prompt assembly.
        """
        return ExampleSelector.select_examples(
            candidate_pool=candidate_pool,
            query_record=query_record,
            num_examples=num_examples,
            diversity=diversity,
            ordering=ordering,
            seed=seed,
        )

    @classmethod
    def render_context_text(cls, demonstrations: List[StandardRecord]) -> str:
        """Renders demonstration records into formatted string block."""
        formatted_demos = [format_single_demonstration(rec) for rec in demonstrations]
        return "\n\n".join(formatted_demos)
