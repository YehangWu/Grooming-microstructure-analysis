# -*- coding: utf-8 -*-
"""
Core rules:
1) Round (bout) assignment:
   - If time gap between consecutive events is > gap_s -> new round
   - Else (<= gap_s) -> same round

2) Background insertion (grooming=0):
   - Within the same round: insert background segments for gaps in (bg_gap_low, bg_gap_high)
   - Across rounds / long gaps: insert background segments for gaps >= bg_gap_high
     (background segment is assigned to the previous round_id)

3) Transition statistics:
   - For each round, build a label sequence (including inserted background 0 segments),
     then force wrap with zeros: [0] + seq + [0]
   - Compress consecutive duplicates, then count adjacent changes as transitions
   - Allowed transitions default: (0->1,1->2,2->3,3->4,4->5,5->0)

Outputs:
- One Excel file per input txt: <stem>_analysis.xlsx
- Multiple sheets containing details and summaries.

Recommended usage (Windows example):
python grooming_batch_analysis.py ^
  --input-dir "input_path" ^
  --gap-s 6.0 --bg-gap-low 0.05 --bg-gap-high 6.0 ^
  --out-dir "out_path" ^
  --allowed "0,1;1,2;2,3;3,4;4,5;5,0"

Dependencies:
- pandas, numpy
- Excel writer engine: xlsxwriter (recommended) or openpyxl
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Column schema (input headers are Chinese)
# -----------------------------
ID_COL = "ID"
LABEL_COL = "grooming"
START_COL = "start_time"
END_COL = "end_time"
DUR_COL = "duration"

REQUIRED_COLS = [ID_COL, LABEL_COL, START_COL, END_COL, DUR_COL]


# -----------------------------
# Exceptions
# -----------------------------
class DataValidationError(RuntimeError):
    """Raised when input table fails validation."""

    def __init__(self, errors: List[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ValidationConfig:
    duration_tol: float = 1e-6
    require_strictly_increasing_id: bool = True
    tolerant_duration_fix: bool = True


@dataclass(frozen=True)
class AnalysisConfig:
    gap_s: float = 6.0
    bg_gap_low: float = 0.05
    bg_gap_high: float = 6.0
    allowed_transitions: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
    )


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_path: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("grooming_batch_analysis")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -----------------------------
# I/O
# -----------------------------
def read_txt_table(path: Path, encodings: Sequence[str] = ("utf-8-sig", "utf-8", "gbk")) -> pd.DataFrame:
    """
    Read a .txt file that is either tab-delimited or whitespace-delimited with a header row.
    """
    last_errs: List[str] = []
    for enc in encodings:
        try:
            return pd.read_csv(path, sep="\t", encoding=enc)
        except Exception as e:
            last_errs.append(f"[tab/{enc}] {e}")

    for enc in encodings:
        try:
            return pd.read_csv(path, sep=r"\s+", engine="python", encoding=enc)
        except Exception as e:
            last_errs.append(f"[whitespace/{enc}] {e}")

    raise RuntimeError(
        f"Failed to read file: {path}\n"
        "Expected tab- or whitespace-delimited text with a header row.\n"
        + "\n".join(last_errs)
    )


def validate_and_clean(df: pd.DataFrame, vcfg: ValidationConfig) -> pd.DataFrame:
    """
    Validate required columns, numeric types, missing values, time logic,
    and optionally fix duration inconsistencies.
    """
    errors: List[str] = []

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataValidationError([f"Missing required columns: {missing}. Required: {REQUIRED_COLS}"])

    out = df.copy()

    # Coerce numeric types
    try:
        out[ID_COL] = pd.to_numeric(out[ID_COL], errors="raise", downcast="integer")
    except Exception:
        errors.append(f"Column '{ID_COL}' must be integer-like.")

    for col in [LABEL_COL, START_COL, END_COL, DUR_COL]:
        try:
            out[col] = pd.to_numeric(out[col], errors="raise")
        except Exception:
            errors.append(f"Column '{col}' must be numeric.")

    # Missing values
    if out[REQUIRED_COLS].isna().any().any():
        locs = out[REQUIRED_COLS].isna().stack()
        bad_cells = [(int(i), str(c)) for (i, c), v in locs.items() if v]
        errors.append(f"NaN detected (row, col) examples: {bad_cells[:5]} ... total={len(bad_cells)}")

    # Time logic
    bad_order = out[out[START_COL] > out[END_COL]]
    if not bad_order.empty:
        idxs = bad_order.index.tolist()[:5]
        errors.append(f"Found rows with start_time > end_time. Examples (first 5): {idxs}")

    bad_neg = out[out[DUR_COL] < 0]
    if not bad_neg.empty:
        idxs = bad_neg.index.tolist()[:5]
        errors.append(f"Found rows with negative duration. Examples (first 5): {idxs}")

    # Duration consistency
    calc = out[END_COL] - out[START_COL]
    diff = (out[DUR_COL] - calc).abs()
    bad = diff > vcfg.duration_tol
    if bad.any():
        idxs = out.index[bad].tolist()[:5]
        msg = f"Duration != (end-start) beyond tol={vcfg.duration_tol}. Examples (first 5): {idxs}"
        if vcfg.tolerant_duration_fix:
            out.loc[bad, DUR_COL] = calc.loc[bad]
            msg += " (auto-fixed duration using end-start)"
        errors.append(msg)

    # Sort by time (stable)
    out_sorted = out.sort_values(by=[START_COL, END_COL, ID_COL]).reset_index(drop=True)
    if not out_sorted[START_COL].equals(out[START_COL].reset_index(drop=True)):
        # Not an error; just normalize order for reproducibility.
        out = out_sorted
    else:
        out = out_sorted  # still normalize index

    # ID checks
    if out[ID_COL].duplicated().any():
        dups = out.loc[out[ID_COL].duplicated(), ID_COL].tolist()[:5]
        errors.append(f"Duplicated '{ID_COL}' values. Examples (first 5): {dups}")

    if vcfg.require_strictly_increasing_id:
        s = out[ID_COL].to_numpy()
        if len(s) > 1 and not np.all(s[1:] > s[:-1]):
            errors.append(f"Column '{ID_COL}' must be strictly increasing.")

    if errors:
        raise DataValidationError(errors)

    return out


# -----------------------------
# Core computations
# -----------------------------
def assign_rounds(df: pd.DataFrame, gap_s: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add columns:
      - gap_to_prev: start_time - prev_end_time
      - round_id: integer round id starting from 1
    Return:
      - df_marked: event-level table with round annotation
      - round_summary: per-round summary table
    """
    out = df.copy()

    prev_end = out[END_COL].shift(1)
    out["gap_to_prev"] = out[START_COL] - prev_end

    round_ids: List[int] = []
    current = 0
    for i in range(len(out)):
        if i == 0:
            current += 1
        else:
            gap = out.at[i, "gap_to_prev"]
            if pd.isna(gap) or float(gap) > gap_s:
                current += 1
        round_ids.append(current)

    out["round_id"] = round_ids

    round_summary = out.groupby("round_id", as_index=False).agg(
        start_idx=(ID_COL, "min"),
        end_idx=(ID_COL, "max"),
        n_events=(ID_COL, "count"),
        round_start=(START_COL, "min"),
        round_end=(END_COL, "max"),
    )
    round_summary["sum_duration_events"] = out.groupby("round_id")[DUR_COL].sum().values
    round_summary["round_span"] = round_summary["round_end"] - round_summary["round_start"]

    df_marked = out[[ID_COL, LABEL_COL, START_COL, END_COL, DUR_COL, "gap_to_prev", "round_id"]].copy()
    return df_marked, round_summary


def insert_background_segments(
    df_marked: pd.DataFrame,
    bg_gap_low: float,
    bg_gap_high: float,
) -> pd.DataFrame:
    """
    Insert background=0 segments based on time gaps.

    Rules:
    - Same round: insert background for gap in (bg_gap_low, bg_gap_high)
    - gap >= bg_gap_high: insert background and mark as cross_round_background=True
      background segment is assigned to the previous row's round_id
    """
    required = [ID_COL, LABEL_COL, START_COL, END_COL, DUR_COL, "gap_to_prev", "round_id"]
    missing = [c for c in required if c not in df_marked.columns]
    if missing:
        raise ValueError(f"insert_background_segments: missing columns {missing}")

    df = df_marked.reset_index(drop=True).copy()
    rows: List[Dict[str, object]] = []

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        row["cross_round_background"] = False
        rows.append(row)

        if i == len(df) - 1:
            continue

        cur_end = float(row[END_COL])
        next_start = float(df.iloc[i + 1][START_COL])
        gap = next_start - cur_end

        same_round = (df.iloc[i]["round_id"] == df.iloc[i + 1]["round_id"])

        if same_round and (bg_gap_low < gap < bg_gap_high):
            rows.append(
                {
                    ID_COL: None,
                    LABEL_COL: 0,
                    START_COL: cur_end,
                    END_COL: next_start,
                    DUR_COL: gap,
                    "gap_to_prev": gap,
                    "round_id": row["round_id"],
                    "cross_round_background": False,
                }
            )
        elif gap >= bg_gap_high:
            rows.append(
                {
                    ID_COL: None,
                    LABEL_COL: 0,
                    START_COL: cur_end,
                    END_COL: next_start,
                    DUR_COL: gap,
                    "gap_to_prev": gap,
                    "round_id": row["round_id"],  # assign to previous round
                    "cross_round_background": True,
                }
            )

    df_bg = pd.DataFrame(rows).sort_values(by=["round_id", START_COL, END_COL]).reset_index(drop=True)

    # Preserve original id and assign a new row index for output readability
    df_bg["orig_event_id"] = df_bg[ID_COL]
    df_bg["row_id"] = np.arange(1, len(df_bg) + 1)

    # Reorder columns
    cols = ["row_id", "orig_event_id", "round_id", LABEL_COL, START_COL, END_COL, DUR_COL, "cross_round_background"]
    return df_bg[cols].copy()


def compress_consecutive(seq: Sequence[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xi = int(x)
        if last is None or xi != last:
            out.append(xi)
            last = xi
    return out


def extract_transitions_force_wrap_zero(df_with_bg: pd.DataFrame) -> pd.DataFrame:
    """
    For each round:
      labels_sorted = grooming labels within the round (including inserted background 0)
      seq = [0] + labels_sorted + [0]
      compress consecutive duplicates
      each adjacent change -> one transition

    Also annotate whether a transition involves the synthetic leading/trailing 0.
    """
    if df_with_bg.empty:
        return pd.DataFrame(columns=["round_id", "from", "to", "start0_synthetic", "end0_synthetic"])

    records: List[Dict[str, object]] = []

    for rid, g in df_with_bg.groupby("round_id", sort=True):
        g = g.sort_values(by=[START_COL, END_COL]).reset_index(drop=True)
        labels = [int(x) for x in g[LABEL_COL].tolist()]

        seq = [0] + labels + [0]
        seq_c = compress_consecutive(seq)

        for i in range(len(seq_c) - 1):
            a, b = seq_c[i], seq_c[i + 1]
            if a == b:
                continue
            records.append(
                {
                    "round_id": int(rid),
                    "from": int(a),
                    "to": int(b),
                    "start0_synthetic": bool(i == 0 and a == 0),                # 0 -> first
                    "end0_synthetic": bool(i == len(seq_c) - 2 and b == 0),     # last -> 0
                }
            )

    return pd.DataFrame(records, columns=["round_id", "from", "to", "start0_synthetic", "end0_synthetic"])


def transition_stats(transitions: pd.DataFrame, allowed: Iterable[Tuple[int, int]]) -> Dict[str, pd.DataFrame]:
    """
    Compute:
      - type_counts: counts of each transition type
      - summary: total/correct/incorrect and proportions
      - per_round: per-round correct/incorrect proportions
      - edge_summary: counts of transitions involving synthetic start/end 0
    """
    if transitions.empty:
        return {
            "type_counts": pd.DataFrame(columns=["transition", "count"]),
            "summary": pd.DataFrame([{"total": 0, "correct": 0, "incorrect": 0, "p_correct": np.nan, "p_incorrect": np.nan}]),
            "per_round": pd.DataFrame(columns=["round_id", "correct", "incorrect", "total", "p_correct", "p_incorrect"]),
            "edge_summary": pd.DataFrame(columns=["type", "count"]),
        }

    allowed_set = set((int(a), int(b)) for a, b in allowed)

    df = transitions.copy()
    df["transition"] = df.apply(lambda r: f"{int(r['from'])}->{int(r['to'])}", axis=1)
    df["is_correct"] = df.apply(lambda r: (int(r["from"]), int(r["to"])) in allowed_set, axis=1)

    type_counts = df["transition"].value_counts().rename_axis("transition").reset_index(name="count")

    total = int(len(df))
    correct = int(df["is_correct"].sum())
    incorrect = int(total - correct)
    summary = pd.DataFrame(
        [
            {
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "p_correct": correct / total if total else np.nan,
                "p_incorrect": incorrect / total if total else np.nan,
            }
        ]
    )

    per_round = (
        df.groupby("round_id")
        .agg(correct=("is_correct", "sum"), total=("is_correct", "count"))
        .reset_index()
    )
    per_round["incorrect"] = per_round["total"] - per_round["correct"]
    per_round["p_correct"] = per_round["correct"] / per_round["total"]
    per_round["p_incorrect"] = per_round["incorrect"] / per_round["total"]

    edge_summary = pd.DataFrame(
        [
            {"type": "transitions_with_synthetic_start0", "count": int(df["start0_synthetic"].sum())},
            {"type": "transitions_with_synthetic_end0", "count": int(df["end0_synthetic"].sum())},
        ]
    )

    return {
        "type_counts": type_counts,
        "summary": summary,
        "per_round": per_round,
        "edge_summary": edge_summary,
    }


def action_duration_share_excluding_background(df_with_bg: pd.DataFrame) -> pd.DataFrame:
    """
    Total duration per action type (excluding background=0), and fraction over total action time.
    """
    df_act = df_with_bg[df_with_bg[LABEL_COL] != 0].copy()
    if df_act.empty:
        return pd.DataFrame(columns=[LABEL_COL, "total_duration", "fraction"])

    g = df_act.groupby(LABEL_COL)[DUR_COL].sum().reset_index(name="total_duration")
    total = float(g["total_duration"].sum())
    g["fraction"] = g["total_duration"] / total if total > 0 else np.nan
    return g.sort_values(LABEL_COL).reset_index(drop=True)


def round_action_time_share_excluding_background(df_with_bg: pd.DataFrame) -> pd.DataFrame:
    """
    Per-round total action time (excluding background), as a fraction of global total action time.
    """
    df_act = df_with_bg[df_with_bg[LABEL_COL] != 0].copy()
    if df_act.empty:
        return pd.DataFrame(columns=["round_id", "round_action_duration", "fraction_of_global_action_time"])

    g = df_act.groupby("round_id")[DUR_COL].sum().reset_index(name="round_action_duration")
    total = float(g["round_action_duration"].sum())
    g["fraction_of_global_action_time"] = g["round_action_duration"] / total if total > 0 else np.nan
    return g


def round_nonbg_over_total_duration(df_with_bg: pd.DataFrame) -> pd.DataFrame:
    """
    Per-round non-background action time / global total time (including background).
    """
    if df_with_bg.empty:
        return pd.DataFrame(columns=["round_id", "round_nonbg_duration", "global_total_duration", "fraction"])

    global_total = float(df_with_bg[DUR_COL].sum())
    g = (
        df_with_bg[df_with_bg[LABEL_COL] != 0]
        .groupby("round_id")[DUR_COL]
        .sum()
        .reset_index(name="round_nonbg_duration")
    )
    g["global_total_duration"] = global_total
    g["fraction"] = g["round_nonbg_duration"] / global_total if global_total > 0 else np.nan
    return g


def round_span_over_global_span(round_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Per-round span / global span.
    """
    if round_summary.empty:
        return pd.DataFrame(columns=["round_id", "round_span", "global_span", "fraction"])

    global_start = float(round_summary["round_start"].min())
    global_end = float(round_summary["round_end"].max())
    global_span = global_end - global_start

    out = round_summary[["round_id", "round_span"]].copy()
    out["global_span"] = global_span
    out["fraction"] = out["round_span"] / global_span if global_span > 0 else np.nan
    return out


# -----------------------------
# File-level analysis
# -----------------------------
def analyze_one_file(
    path: Path,
    vcfg: ValidationConfig,
    acfg: AnalysisConfig,
) -> Dict[str, pd.DataFrame]:
    raw = read_txt_table(path)
    clean = validate_and_clean(raw, vcfg)

    df_marked, round_summary = assign_rounds(clean, gap_s=acfg.gap_s)
    df_with_bg = insert_background_segments(df_marked, bg_gap_low=acfg.bg_gap_low, bg_gap_high=acfg.bg_gap_high)

    transitions = extract_transitions_force_wrap_zero(df_with_bg)
    tr = transition_stats(transitions, allowed=acfg.allowed_transitions)

    act_share = action_duration_share_excluding_background(df_with_bg)
    round_act_share = round_action_time_share_excluding_background(df_with_bg)
    round_nonbg_total = round_nonbg_over_total_duration(df_with_bg)
    round_span_share = round_span_over_global_span(round_summary)

    # Add a compact "overview" table for the Excel
    overview = pd.DataFrame(
        [
            {"item": "input_file", "value": str(path.resolve())},
            {"item": "n_rounds", "value": int(round_summary["round_id"].nunique()) if not round_summary.empty else 0},
            {"item": "gap_s", "value": acfg.gap_s},
            {"item": "bg_gap_same_round", "value": f"({acfg.bg_gap_low}, {acfg.bg_gap_high})"},
            {"item": "bg_gap_cross_round", "value": f"gap >= {acfg.bg_gap_high}"},
            {"item": "transition_rule", "value": "per-round forced wrap: [0]+seq+[0], compress, count changes"},
            {"item": "n_allowed_transitions", "value": len(acfg.allowed_transitions)},
        ]
    )

    return {
        "events_with_round": df_marked,
        "events_with_bg0": df_with_bg,
        "round_summary": round_summary,
        "transitions": transitions,
        "transition_type_counts": tr["type_counts"],
        "transition_summary": tr["summary"],
        "transition_per_round": tr["per_round"],
        "transition_edge_summary": tr["edge_summary"],
        "action_duration_share_no_bg": act_share,
        "round_action_time_share_no_bg": round_act_share,
        "round_nonbg_over_total": round_nonbg_total,
        "round_span_over_global": round_span_share,
        "overview": overview,
    }


def write_excel(outputs: Dict[str, pd.DataFrame], out_xlsx: Path, engine: str = "xlsxwriter") -> None:
    """
    Write all outputs into a multi-sheet Excel file.
    """
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    sheet_order = [
        ("overview", "Overview"),
        ("events_with_round", "EventTable_Rounds"),
        ("events_with_bg0", "EventTable_WithBG0"),
        ("round_summary", "RoundSummary"),
        ("transitions", "Transitions_Detail"),
        ("transition_type_counts", "Transitions_TypeCounts"),
        ("transition_summary", "Transitions_Summary"),
        ("transition_per_round", "Transitions_PerRound"),
        ("transition_edge_summary", "Transitions_Edge0"),
        ("action_duration_share_no_bg", "ActionDurationShare_NoBG"),
        ("round_action_time_share_no_bg", "RoundActionShare_NoBG"),
        ("round_nonbg_over_total", "RoundNonBG_over_TotalTime"),
        ("round_span_over_global", "RoundSpan_over_GlobalSpan"),
    ]

    with pd.ExcelWriter(out_xlsx, engine=engine) as writer:
        for key, sheet in sheet_order:
            df = outputs.get(key)
            if df is None or df.empty:
                # Still write an empty sheet for consistency
                pd.DataFrame().to_excel(writer, sheet_name=sheet, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet, index=False)


# -----------------------------
# CLI helpers
# -----------------------------
def parse_allowed_transitions(text: str) -> Tuple[Tuple[int, int], ...]:
    """
    Parse transitions like: "0,1;1,2;2,3;3,4;4,5;5,0"
    """
    pairs: List[Tuple[int, int]] = []
    text = text.strip()
    if not text:
        return tuple()
    for chunk in text.split(";"):
        a_str, b_str = chunk.split(",")
        pairs.append((int(a_str.strip()), int(b_str.strip())))
    return tuple(pairs)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch grooming analysis with background insertion and transition stats.")
    p.add_argument("--input-dir", required=True, type=str, help="Directory containing .txt files.")
    p.add_argument("--out-dir", default=None, type=str, help="Output directory (default: <input-dir>/out).")
    p.add_argument("--gap-s", default=6.0, type=float, help="Round split threshold (gap > gap_s -> new round).")
    p.add_argument("--bg-gap-low", default=0.05, type=float, help="Lower bound for inserting background within a round.")
    p.add_argument("--bg-gap-high", default=6.0, type=float, help="Upper bound within round; also cross-round bg if gap>=.")
    p.add_argument(
        "--allowed",
        default="0,1;1,2;2,3;3,4;4,5;5,0",
        type=str,
        help="Allowed transitions, e.g. '0,1;1,2;2,3;3,4;4,5;5,0'.",
    )
    p.add_argument("--encoding", default=None, type=str, help="Optional preferred encoding (utf-8-sig/utf-8/gbk).")
    p.add_argument("--excel-engine", default="xlsxwriter", type=str, help="xlsxwriter (recommended) or openpyxl.")
    p.add_argument("--log", default=None, type=str, help="Optional log file path.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (input_dir / "out")
    logger = setup_logger(Path(args.log) if args.log else None)

    allowed = parse_allowed_transitions(args.allowed)
    vcfg = ValidationConfig(duration_tol=1e-6, require_strictly_increasing_id=True, tolerant_duration_fix=True)
    acfg = AnalysisConfig(
        gap_s=float(args.gap_s),
        bg_gap_low=float(args.bg_gap_low),
        bg_gap_high=float(args.bg_gap_high),
        allowed_transitions=allowed if allowed else AnalysisConfig().allowed_transitions,
    )

    txt_files = sorted([p for p in input_dir.glob("*.txt") if p.is_file()])
    if not txt_files:
        logger.warning(f"No .txt files found under: {input_dir.resolve()}")
        return

    logger.info(f"Found {len(txt_files)} txt files under: {input_dir.resolve()}")
    logger.info(f"Output directory: {out_dir.resolve()}")

    for txt in txt_files:
        logger.info(f"Processing: {txt.name}")
        try:
            outputs = analyze_one_file(txt, vcfg=vcfg, acfg=acfg)
            out_xlsx = out_dir / f"{txt.stem}_analysis.xlsx"
            write_excel(outputs, out_xlsx, engine=args.excel_engine)
            logger.info(f"Saved: {out_xlsx.resolve()}")
        except DataValidationError as e:
            logger.error(f"Validation failed for {txt.name}:\n{e}")
        except Exception as e:
            logger.exception(f"Failed for {txt.name}: {e}")


if __name__ == "__main__":
    main()
