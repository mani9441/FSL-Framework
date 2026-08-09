"""
Dissertation Artifacts Generator for Phase C & D.
Aggregates experimental campaign results using task-aware metric schemas and produces Chapter 4 & 5 research artifacts.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import json

from evaluation.registry import TaskMetricRegistry
from evaluation import ClassificationEvaluator, QAEvaluator, GenerationEvaluator
from reports.latex_generator import LaTeXTableGenerator
from response_repository import ResponseRepository
from utilities.constants import RESULTS_DIR
from utilities.helpers import save_json
from utilities.logger import get_logger

logger = get_logger("dissertation_generator")

DISSERTATION_DIR = Path(__file__).resolve().parent.parent / "dissertation"


class DissertationArtifactGenerator:
    """Generates task-aware publication-ready tables, LaTeX source code, and observations for Chapters 4 & 5."""

    def __init__(self, dissertation_dir: Path = DISSERTATION_DIR):
        self.dissertation_dir = Path(dissertation_dir)
        self.dissertation_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_artifacts(self, responses_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Loads response records across completed experiment folders and produces Chapters 4 & 5 task-aware research artifacts.
        """
        logger.info("==================================================")
        logger.info("Generating Phase C & D Dissertation Research Artifacts")
        logger.info("==================================================")

        from utilities.constants import RUNS_DIR
        responses_base = responses_dir or (self.dissertation_dir.parent / "experiments")
        if not responses_base.exists():
            responses_base = RUNS_DIR

        all_records = ResponseRepository.load_all_responses(responses_base)

        if not all_records:
            logger.warning(f"No response records found in {responses_base}.")

        df = ResponseRepository.to_dataframe(all_records)
        if df.empty:
            logger.error("Response DataFrame is empty. Cannot generate dissertation tables.")
            return {}

        # 1. Master CSV Export
        master_csv_path = self.dissertation_dir / "master_campaign_dataset.csv"
        df.to_csv(master_csv_path, index=False, encoding="utf-8")
        logger.info(f"Exported master campaign dataset ({len(df)} records) to '{master_csv_path}'")

        # Add metric column for pivot scaling matrix if absent
        df["primary_performance_metric"] = None
        for idx, row in df.iterrows():
            if row.get("success") is False:
                df.at[idx, "primary_performance_metric"] = None
                continue
            t_type = str(row.get("task_type", "text_classification")).lower()
            if t_type == "text_generation":
                df.at[idx, "primary_performance_metric"] = row.get("rouge_l", row.get("bleu_4"))
            elif t_type == "question_answering":
                df.at[idx, "primary_performance_metric"] = row.get("response_correctness", row.get("qa_f1"))
            else:
                is_c = row.get("is_correct")
                if is_c is True:
                    df.at[idx, "primary_performance_metric"] = 1.0
                elif is_c is False:
                    df.at[idx, "primary_performance_metric"] = 0.0
                else:
                    df.at[idx, "primary_performance_metric"] = None

        # 2. Table 4.1: Model & Task vs Few-Shot Demonstration Scaling Matrix
        index_cols = ["task_type", "model_name"] if "task_type" in df.columns else ["model_name"]
        pivot_df = df.pivot_table(
            index=index_cols,
            columns="num_examples",
            values="primary_performance_metric",
            aggfunc="mean"
        ).reset_index()

        col_map = {0: "0-shot", 1: "1-shot", 3: "3-shot", 5: "5-shot"}
        pivot_df = pivot_df.rename(columns=col_map)

        t41_tex = LaTeXTableGenerator.generate_latex_table(
            pivot_df,
            caption="Table 4.1: Task-Aware Few-Shot Performance Scaling Matrix (0-shot, 1-shot, 3-shot, 5-shot)",
            label="tab:few_shot_scaling_matrix",
        )
        (self.dissertation_dir / "table_4_1_few_shot_scaling_matrix.tex").write_text(t41_tex, encoding="utf-8")
        pivot_df.to_csv(self.dissertation_dir / "table_4_1_few_shot_scaling_matrix.csv", index=False)

        # 3. Table 4.2: Task-Aware Shot Count Performance & Efficiency Breakdown
        rows = []
        group_cols = ["task_type", "num_examples"] if "task_type" in df.columns else ["num_examples"]

        for g_keys, group in df.groupby(group_cols):
            if isinstance(g_keys, tuple):
                t_type, k_shot = g_keys
            else:
                t_type, k_shot = "text_classification", g_keys

            preds = group["parsed_prediction"].fillna("").tolist() if "parsed_prediction" in group.columns else group["generated_text"].fillna("").tolist()
            refs = group["ground_truth"].fillna("").tolist()
            parse_succs = group["parse_success"].tolist() if "parse_success" in group.columns else [True] * len(group)
            req_succs = group["success"].tolist() if "success" in group.columns else [True] * len(group)

            succ_group = group[group["success"] == True] if "success" in group.columns else group
            if not succ_group.empty:
                mean_lat = round(succ_group["latency_seconds"].mean(), 4) if "latency_seconds" in succ_group.columns else None
                mean_p_tok = round(succ_group["prompt_tokens"].mean(), 1) if "prompt_tokens" in succ_group.columns else None
                mean_c_tok = round(succ_group["completion_tokens"].mean(), 1) if "completion_tokens" in succ_group.columns else None
                mean_t_tok = round(succ_group["total_tokens"].mean(), 1) if "total_tokens" in succ_group.columns else None
            else:
                mean_lat = None
                mean_p_tok = None
                mean_c_tok = None
                mean_t_tok = None

            row = {
                "task_type": t_type,
                "num_examples": k_shot,
                "sample_count": len(group),
                "request_success_rate": round(group["success"].mean(), 4) if "success" in group.columns else 1.0,
                "parse_success_rate": round(group["parse_success"].mean(), 4) if "parse_success" in group.columns else 1.0,
                "latency_seconds": mean_lat,
                "prompt_tokens": mean_p_tok,
                "completion_tokens": mean_c_tok,
                "total_tokens": mean_t_tok,
            }

            if t_type == "text_classification":
                perf = ClassificationEvaluator.evaluate(preds, refs, parse_successes=parse_succs, request_successes=req_succs)
                row["accuracy"] = perf.get("accuracy")
                row["f1"] = perf.get("f1_score")
            elif t_type == "question_answering":
                contexts = group["prompt_text"].tolist() if "prompt_text" in group.columns else None
                perf = QAEvaluator.evaluate(preds, refs, contexts=contexts, request_successes=req_succs)
                row["response_correctness"] = perf.get("response_correctness")
                row["response_completeness"] = perf.get("response_completeness")
                row["response_relevance"] = perf.get("response_relevance")
            elif t_type == "text_generation":
                perf = GenerationEvaluator.evaluate(preds, refs, request_successes=req_succs)
                row["bleu_4"] = perf.get("bleu_4")
                row["rouge_1"] = perf.get("rouge_1")
                row["rouge_l"] = perf.get("rouge_l")
            else:
                perf = ClassificationEvaluator.evaluate(preds, refs, parse_successes=parse_succs, request_successes=req_succs)
                row["accuracy"] = perf.get("accuracy")

            rows.append(row)

        shot_group = pd.DataFrame(rows)

        t42_tex = LaTeXTableGenerator.generate_latex_table(
            shot_group,
            caption="Table 4.2: Task-Aware Few-Shot Demonstration Scaling Breakdown (0-shot vs 1-shot vs 3-shot vs 5-shot)",
            label="tab:shot_count_comparison",
        )
        (self.dissertation_dir / "table_4_2_shot_count_comparison.tex").write_text(t42_tex, encoding="utf-8")
        shot_group.to_csv(self.dissertation_dir / "table_4_2_shot_count_comparison.csv", index=False)

        # 4. Research Observations Report
        obs_lines = [
            "# Chapter 4 & 5 Research Analysis & Key Observations",
            "**Generated**: Automated Task-Aware Dissertation Artifact Engine",
            "",
            "## 1. Task-Aware Model Evaluation (Table 4.1)",
            f"- Analyzed **{len(df)}** response records across task types: {', '.join(df['task_type'].unique().tolist()) if 'task_type' in df.columns else 'default'}.",
            "",
            "## 2. Task Metric Breakdown (Table 4.2)",
        ]

        if "task_type" in df.columns:
            for t in df["task_type"].unique():
                sub_df = df[df["task_type"] == t]
                if t == "text_classification":
                    acc = sub_df["is_correct"].mean() if "is_correct" in sub_df.columns else 0.0
                    obs_lines.append(f"- **Text Classification**: Overall Mean Accuracy = `{acc:.2%}`.")
                elif t == "question_answering":
                    corr = sub_df["response_correctness"].mean() if "response_correctness" in sub_df.columns else 0.0
                    obs_lines.append(f"- **Question Answering**: Overall Response Correctness = `{corr:.4f}`.")
                elif t == "text_generation":
                    r_l = sub_df["rouge_l"].mean() if "rouge_l" in sub_df.columns else 0.0
                    b_4 = sub_df["bleu_4"].mean() if "bleu_4" in sub_df.columns else 0.0
                    obs_lines.append(f"- **Text Generation**: Overall ROUGE-L = `{r_l:.4f}`, BLEU-4 = `{b_4:.4f}`.")

        obs_lines.extend([
            "",
            "---",
            "*All tables exported in LaTeX booktabs and CSV formats under dissertation/*",
        ])
        obs_path = self.dissertation_dir / "research_observations_report.md"
        obs_path.write_text("\n".join(obs_lines) + "\n", encoding="utf-8")

        # 5. Final Summary JSON
        unique_models = df["model_name"].unique().tolist() if "model_name" in df.columns else []
        shot_counts = df["num_examples"].unique().tolist() if "num_examples" in df.columns else []

        summary_payload = {
            "total_records_analyzed": len(df),
            "unique_models": unique_models,
            "shot_counts": shot_counts,
            "tasks_analyzed": df["task_type"].unique().tolist() if "task_type" in df.columns else [],
            "tables": {
                "table_4_1": str(self.dissertation_dir / "table_4_1_few_shot_scaling_matrix.tex"),
                "table_4_2": str(self.dissertation_dir / "table_4_2_shot_count_comparison.tex"),
            },
        }
        save_json(summary_payload, self.dissertation_dir / "final_experiment_summary.json")

        logger.info("==================================================")
        logger.info(f"Dissertation Artifacts Generated at '{self.dissertation_dir}'")
        logger.info("==================================================")

        return {
            "table_4_1": self.dissertation_dir / "table_4_1_few_shot_scaling_matrix.tex",
            "table_4_2": self.dissertation_dir / "table_4_2_shot_count_comparison.tex",
            "observations": obs_path,
            "summary_json": self.dissertation_dir / "final_experiment_summary.json",
        }
