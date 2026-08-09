"""
Statistical Significance Tester for FSL Research Framework.
Executes T-tests, ANOVA, Mann-Whitney U tests, and computes Cohen's d effect sizes.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats

from analysis.schema import StatisticalTestResult
from utilities.logger import get_logger

logger = get_logger("significance_tester")


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculates Cohen's d effect size between two sample groups."""
    if not group1 or not group2 or len(group1) < 2 or len(group2) < 2:
        return 0.0

    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if s_pooled == 0.0:
        return 0.0

    return float((m1 - m2) / s_pooled)


class SignificanceTester:
    """Significance testing engine for hypothesis evaluation."""

    @classmethod
    def two_sample_ttest(
        cls,
        group1: List[float],
        group2: List[float],
        paired: bool = False,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """
        Executes Independent or Paired Two-Sample T-Test.
        """
        test_name = "Paired T-Test" if paired else "Independent T-Test"

        if not group1 or not group2 or len(group1) < 2 or len(group2) < 2:
            return StatisticalTestResult(
                test_name=test_name,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                effect_size_cohens_d=0.0,
                interpretation="Insufficient sample size for T-Test.",
            )

        if paired and len(group1) == len(group2):
            res = stats.ttest_rel(group1, group2)
        else:
            res = stats.ttest_ind(group1, group2, equal_var=False)

        stat_val = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
        is_sig = p_val < alpha
        cohens_d = compute_cohens_d(group1, group2)

        interp = (
            f"Statistically SIGNIFICANT difference (p={p_val:.6f} < {alpha}, Cohen's d={cohens_d:.2f})."
            if is_sig
            else f"No statistically significant difference (p={p_val:.6f} >= {alpha})."
        )

        logger.info(f"{test_name}: p={p_val:.6f}, significant={is_sig}, d={cohens_d:.2f}")
        return StatisticalTestResult(
            test_name=test_name,
            statistic=stat_val,
            p_value=p_val,
            is_significant=is_sig,
            effect_size_cohens_d=cohens_d,
            interpretation=interp,
        )

    @classmethod
    def one_way_anova(
        cls, groups: List[List[float]], alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Executes One-Way ANOVA across multiple sample groups.
        """
        test_name = "One-Way ANOVA"
        valid_groups = [g for g in groups if len(g) >= 2]

        if len(valid_groups) < 2:
            return StatisticalTestResult(
                test_name=test_name,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                interpretation="ANOVA requires at least 2 valid sample groups.",
            )

        res = stats.f_oneway(*valid_groups)
        stat_val = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
        is_sig = p_val < alpha

        interp = (
            f"Statistically SIGNIFICANT variance across groups (p={p_val:.6f} < {alpha})."
            if is_sig
            else f"No statistically significant difference across groups (p={p_val:.6f} >= {alpha})."
        )

        logger.info(f"{test_name}: F={stat_val:.4f}, p={p_val:.6f}, significant={is_sig}")
        return StatisticalTestResult(
            test_name=test_name,
            statistic=stat_val,
            p_value=p_val,
            is_significant=is_sig,
            effect_size_cohens_d=0.0,
            interpretation=interp,
        )

    @classmethod
    def mann_whitney_u(
        cls, group1: List[float], group2: List[float], alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Executes Mann-Whitney U non-parametric test.
        """
        test_name = "Mann-Whitney U Test"

        if not group1 or not group2 or len(group1) < 2 or len(group2) < 2:
            return StatisticalTestResult(
                test_name=test_name,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                interpretation="Insufficient sample size for Mann-Whitney U test.",
            )

        res = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        stat_val = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
        is_sig = p_val < alpha
        cohens_d = compute_cohens_d(group1, group2)

        interp = (
            f"Statistically SIGNIFICANT non-parametric difference (p={p_val:.6f} < {alpha})."
            if is_sig
            else f"No statistically significant difference (p={p_val:.6f} >= {alpha})."
        )

        logger.info(f"{test_name}: U={stat_val:.4f}, p={p_val:.6f}, significant={is_sig}")
        return StatisticalTestResult(
            test_name=test_name,
            statistic=stat_val,
            p_value=p_val,
            is_significant=is_sig,
            effect_size_cohens_d=cohens_d,
            interpretation=interp,
        )
