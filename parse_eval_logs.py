"""
parse_eval_logs.py

Parses eval log files containing chunks like:
    ── Eval Results for 'variant_name' ──
      Model    : FiLM+RWSE+GCN  (or Baseline)
      Episodes : 860
      Avg len  : 115.8 ± 48.9 steps
      ...

And individual episode lines:
    Episode N: len=X  (N/total)

Computes: mean, std, median, min, max, top-10% mean, 95% bootstrapped CI,
          % episodes > 200 steps, % episodes > 500 steps.

Usage:
    python parse_eval_logs.py --log film_eval.log --out results_film.json
    python parse_eval_logs.py --log baseline_eval.log --out results_baseline.json
    python parse_eval_logs.py --log film.log --log baseline.log --compare
"""

import re
import json
import argparse
import numpy as np
from collections import defaultdict


def bootstrap_ci(data, n_boot=10000, ci=95, seed=42):
    rng     = np.random.default_rng(seed)
    n       = len(data)
    means   = [np.mean(rng.choice(data, size=n, replace=True))
               for _ in range(n_boot)]
    lo      = np.percentile(means, (100 - ci) / 2)
    hi      = np.percentile(means, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def bootstrap_ci_topk(data, k, n_boot=10000, ci=95, seed=42):
    rng     = np.random.default_rng(seed)
    n       = len(data)
    means   = [np.mean(np.sort(rng.choice(data, size=n, replace=True))[-k:])
               for _ in range(n_boot)]
    lo      = np.percentile(means, (100 - ci) / 2)
    hi      = np.percentile(means, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def compute_metrics(episode_lengths):
    data     = np.array(episode_lengths, dtype=float)
    n        = len(data)
    mean     = float(np.mean(data))
    std      = float(np.std(data))
    median   = float(np.median(data))
    min_ep   = int(np.min(data))
    max_ep   = int(np.max(data))

    # 95% bootstrapped CI on mean
    ci_lo, ci_hi = bootstrap_ci(data)

    # Top 10% mean + CI
    k             = max(1, n // 20)
    top_k_vals    = np.sort(data)[-k:]
    top_k_mean    = float(np.mean(top_k_vals))
    tk_lo, tk_hi  = bootstrap_ci_topk(data, k)

    # Success rates
    pct_200  = float(np.mean(data > 200) * 100)
    pct_500  = float(np.mean(data > 500) * 100)

    return {
        "n":            n,
        "mean":         round(mean,  1),
        "std":          round(std,   1),
        "median":       round(median,1),
        "min":          min_ep,
        "max":          max_ep,
        "ci_95_lo":     round(ci_lo, 1),
        "ci_95_hi":     round(ci_hi, 1),
        "ci_95_pm":     round((ci_hi - ci_lo) / 2, 1),
        "top10_mean":   round(top_k_mean, 1),
        "top10_ci_lo":  round(tk_lo, 1),
        "top10_ci_hi":  round(tk_hi, 1),
        "top10_ci_pm":  round((tk_hi - tk_lo) / 2, 1),
        "pct_gt_200":   round(pct_200, 1),
        "pct_gt_500":   round(pct_500, 1),
    }


def parse_log(log_path):
    """
    Parse a log file and return dict:
        { variant_name: { model: str, episodes: [int], metrics: dict } }
    """
    with open(log_path) as f:
        content = f.read()

    results = {}

    # Split into chunks by the separator line
    # Each chunk starts with ── Eval Results for '...' ──
    chunk_pattern = re.compile(
        r"── Eval Results for '([^']+)' ──(.*?)(?=── Eval Results for |── Recording|\Z)",
        re.DOTALL
    )

    # Also collect episode lines before each results block
    # Episode lines pattern: Episode N: len=X  (N/total)
    ep_pattern = re.compile(r"Episode \d+: len=(\d+)")

    # Find all result blocks with their positions
    result_blocks = list(chunk_pattern.finditer(content))

    for i, match in enumerate(result_blocks):
        variant_name = match.group(1)
        block        = match.group(2)

        # Extract model type
        model_match = re.search(r"Model\s*:\s*(.+)", block)
        model       = model_match.group(1).strip() if model_match else "Unknown"
        is_baseline = "Baseline" in model

        # Extract reported stats from the block
        avg_match = re.search(r"Avg len\s*:\s*([\d.]+)\s*±\s*([\d.]+)", block)
        med_match = re.search(r"Median\s*:\s*([\d.]+)", block)
        mm_match  = re.search(r"Min/Max\s*:\s*(\d+)\s*/\s*(\d+)", block)

        # Find episode lines BEFORE this result block (they appear in the log before the summary)
        # Search backwards from this match start
        block_start = match.start()
        # Find the previous result block end or log start
        prev_end = result_blocks[i-1].end() if i > 0 else 0
        preceding_text = content[prev_end:block_start]
        episode_lengths = [int(x) for x in ep_pattern.findall(preceding_text)]

        # Also check inside the block itself
        episode_lengths += [int(x) for x in ep_pattern.findall(block)]

        key = f"{variant_name}__{'baseline' if is_baseline else 'film'}"

        entry = {
            "variant_name": variant_name,
            "model":        model,
            "is_baseline":  is_baseline,
            "reported": {
                "avg":    float(avg_match.group(1)) if avg_match else None,
                "std":    float(avg_match.group(2)) if avg_match else None,
                "median": float(med_match.group(1)) if med_match else None,
                "min":    int(mm_match.group(1))    if mm_match  else None,
                "max":    int(mm_match.group(2))    if mm_match  else None,
            },
            "episodes": episode_lengths,
        }

        if episode_lengths:
            entry["metrics"] = compute_metrics(episode_lengths)
        else:
            # Fall back to reported stats if no episode lines found
            entry["metrics"] = None
            print(f"  [Warning] No episode lines found for {variant_name} "
                  f"({'baseline' if is_baseline else 'film'}) — using reported stats only")

        results[key] = entry

    return results


def print_table(results, compare=False):
    """Print a clean comparison table with Top 10% CI."""
    # Group by variant
    by_variant = defaultdict(dict)
    for key, entry in results.items():
        vname = entry["variant_name"]
        mtype = "baseline" if entry["is_baseline"] else "film"
        by_variant[vname][mtype] = entry

    # Header updated to include ± for Top 10%
    print(f"\n{'='*105}")
    print(f"{'Variant':<30} {'Model':<10} {'Mean':>6} {'±CI':>6} "
          f"{'Median':>7} {'Max':>6} {'Top 10%':>10} {'±CI':>6} {'>200%':>7} {'>500%':>7}")
    print(f"{'-'*105}")

    for vname in sorted(by_variant.keys()):
        for mtype in ["film", "baseline"]:
            entry = by_variant[vname].get(mtype)
            if entry is None:
                continue
            m = entry.get("metrics")
            if m:
                # Accessing the 'top10_ci_pm' calculated in compute_metrics
                print(f"{vname:<30} {mtype:<10} "
                      f"{m['mean']:>6.1f} "
                      f"{m['ci_95_pm']:>6.1f} "
                      f"{m['median']:>7.1f} "
                      f"{m['max']:>6} "
                      f"{m['top10_mean']:>10.1f} "
                      f"{m['top10_ci_pm']:>6.1f} "
                      f"{m['pct_gt_200']:>6.1f}% "
                      f"{m['pct_gt_500']:>6.1f}%")
            else:
                r = entry["reported"]
                print(f"{vname:<30} {mtype:<10} "
                      f"{r['avg'] or 'N/A':>6} "
                      f"{'N/A':>6} "
                      f"{r['median'] or 'N/A':>7} "
                      f"{r['max'] or 'N/A':>6} "
                      f"{'N/A':>10} {'N/A':>6} {'N/A':>7} {'N/A':>7}")

    print(f"{'='*105}")

    if compare:
        print(f"\n{'='*80}")
        print("FiLM vs Baseline comparison (FiLM - Baseline):")
        print(f"{'-'*80}")
        for vname in sorted(by_variant.keys()):
            film_entry = by_variant[vname].get("film")
            base_entry = by_variant[vname].get("baseline")
            if film_entry and base_entry:
                fm = film_entry.get("metrics")
                bm = base_entry.get("metrics")
                if fm and bm:
                    diff_mean   = fm["mean"]      - bm["mean"]
                    diff_max    = fm["max"]       - bm["max"]
                    diff_top10  = fm["top10_mean"] - bm["top10_mean"]
                    # Calculate how the CI changed (did the tail become more/less stable?)
                    diff_tk_ci  = fm["top10_ci_pm"] - bm["top10_ci_pm"]
                    diff_pct500 = fm["pct_gt_500"] - bm["pct_gt_500"]
                    winner = "FiLM" if diff_mean > 0 else "Base"
                    
                    print(f"{vname:<30} mean {diff_mean:+.1f}  "
                          f"max {diff_max:+d}  "
                          f"top10% {diff_top10:+.1f} (ΔCI {diff_tk_ci:+.1f})  "
                          f">500% {diff_pct500:+.1f}pp  "
                          f"[{winner}]")
        print(f"{'='*80}")

    if compare:
        print(f"\n{'='*60}")
        print("FiLM vs Baseline comparison (FiLM - Baseline):")
        print(f"{'-'*60}")
        for vname in sorted(by_variant.keys()):
            film_entry = by_variant[vname].get("film")
            base_entry = by_variant[vname].get("baseline")
            if film_entry and base_entry:
                fm = film_entry.get("metrics")
                bm = base_entry.get("metrics")
                if fm and bm:
                    diff_mean   = fm["mean"]     - bm["mean"]
                    diff_max    = fm["max"]       - bm["max"]
                    diff_top10  = fm["top10_mean"]- bm["top10_mean"]
                    diff_pct500 = fm["pct_gt_500"]- bm["pct_gt_500"]
                    winner = "FiLM" if diff_mean > 0 else "Base"
                    print(f"{vname:<30} mean {diff_mean:+.1f}  "
                          f"max {diff_max:+d}  "
                          f"top10% {diff_top10:+.1f}  "
                          f">500% {diff_pct500:+.1f}pp  "
                          f"[{winner}]")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",     nargs="+", required=True,
                        help="Log file(s) to parse")
    parser.add_argument("--out",     default=None,
                        help="Output JSON path")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Print FiLM vs baseline comparison")
    args = parser.parse_args()

    all_results = {}
    for log_path in args.log:
        print(f"Parsing: {log_path}")
        results = parse_log(log_path)
        print(f"  Found {len(results)} result blocks")
        all_results.update(results)

    print_table(all_results, compare=args.compare)

    if args.out:
        # Convert to serializable format
        out = {}
        for key, entry in all_results.items():
            out[key] = {k: v for k, v in entry.items() if k != "episodes"}
            out[key]["n_episodes"] = len(entry.get("episodes", []))
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()