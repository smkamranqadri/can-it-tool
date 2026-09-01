"""Terminal rendering of a model comparison."""

from __future__ import annotations

from ..comparison import DIMENSION_HIGHLIGHTS, HEADLINE_METRICS


def _fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind in {"rate", "rate_lower_better"}:
        return f"{value * 100:.1f}%"
    if kind == "score":
        return f"{value:.3f}"
    return f"{value:.2f}"


def _column_width(entries) -> int:
    return max(12, max(len(e["label"]) for e in entries) + 2)


def render_comparison(comparison: dict) -> str:
    entries = comparison["ranked"]
    if not entries:
        return "No result files to compare."

    width = _column_width(entries)
    lines = ["=" * (26 + width * len(entries))]
    lines.append("MODEL COMPARISON")
    lines.append("=" * (26 + width * len(entries)))

    if comparison["warnings"]:
        lines.append("")
        lines.append("WARNING: these runs are not fully comparable")
        for warning in comparison["warnings"]:
            lines.append(f"  ! {warning}")

    lines.append("")
    header = f"  {'':<24}" + "".join(f"{e['label']:>{width}}" for e in entries)
    lines.append(header)
    lines.append(f"  {'-' * (24 + width * len(entries))}")

    for row in (
        ("runtime", lambda e: e["runtime"] or "-"),
        ("quantization", lambda e: e["quantization"] or "-"),
        ("ctx requested", lambda e: str(e["num_ctx_requested"] or "-")),
        ("runs/scenario", lambda e: str(e["runs_per_scenario"])),
        ("scenarios", lambda e: str(e["scenario_count"])),
    ):
        label, getter = row
        lines.append(f"  {label:<24}" + "".join(f"{getter(e):>{width}}" for e in entries))

    lines.append("")
    for key, label, kind in HEADLINE_METRICS:
        cells = "".join(
            f"{_fmt(e['metrics'].get(key), kind):>{width}}" for e in entries
        )
        marker = " *" if key == "safety_failure_rate" else "  "
        lines.append(f"{marker}{label:<24}{cells}")

    lines.append("")
    for name in DIMENSION_HIGHLIGHTS:
        cells = "".join(f"{_fmt(e['dimensions'].get(name), 'score'):>{width}}" for e in entries)
        lines.append(f"  {name:<24}{cells}")

    lines.append("")
    lines.append(
        f"  {'latency mean ms':<24}"
        + "".join(f"{_fmt(e['latency_mean_ms'], 'number'):>{width}}" for e in entries)
    )
    lines.append(
        f"  {'latency p95 ms':<24}"
        + "".join(f"{_fmt(e['latency_p95_ms'], 'number'):>{width}}" for e in entries)
    )
    lines.append(
        f"  {'gen tok/s (server)':<24}"
        + "".join(
            f"{_fmt(e['generation_tokens_per_second'], 'number'):>{width}}" for e in entries
        )
    )
    lines.append(
        f"  {'tok/s (wall clock)':<24}"
        + "".join(
            f"{_fmt(e['wall_clock_tokens_per_second'], 'number'):>{width}}" for e in entries
        )
    )

    categories = sorted({c for e in entries for c in e["categories"]})
    lines.append("")
    lines.append("CATEGORY SCORES")
    lines.append(f"  {'-' * (24 + width * len(entries))}")
    for category in categories:
        cells = "".join(
            f"{_fmt((e['categories'].get(category) or {}).get('mean_final_score'), 'score'):>{width}}"
            for e in entries
        )
        lines.append(f"  {category:<24}{cells}")

    unsafe = [e for e in entries if not e["safe"]]
    if unsafe:
        lines.append("")
        lines.append("SAFETY VIOLATIONS BY KIND")
        lines.append(f"  {'-' * 60}")
        for entry in unsafe:
            lines.append(f"  {entry['label']}")
            for kind, count in sorted(entry["safety_violation_counts"].items()):
                lines.append(f"      {kind:<44}{count:>4}")

    lines.append("")
    lines.append(f"REGRESSIONS AGAINST {comparison['baseline']}")
    lines.append(f"  {'-' * 60}")
    any_regression = False
    for block in comparison["regressions"]:
        drops = block["metrics"] + block["categories"] + block["scenarios"]
        if not drops:
            lines.append(f"  {block['candidate']}: no regressions")
            continue
        any_regression = True
        lines.append(f"  {block['candidate']}:")
        for drop in block["metrics"]:
            lines.append(
                f"      metric   {drop['label']:<26}"
                f"{drop['before']:.3f} -> {drop['after']:.3f}  ({drop['delta']:+.3f})"
            )
        for drop in block["categories"]:
            lines.append(
                f"      category {drop['category']:<26}"
                f"{drop['before']:.3f} -> {drop['after']:.3f}  ({drop['delta']:+.3f})"
            )
        for drop in block["scenarios"][:10]:
            flag = "NEWLY UNSAFE" if drop["newly_unsafe"] else "now failing "
            lines.append(
                f"      {flag} {drop['scenario']:<38}{drop['before']} -> {drop['after']}"
            )
        if len(block["scenarios"]) > 10:
            lines.append(f"      ... and {len(block['scenarios']) - 10} more scenarios")
    if not any_regression and len(comparison["regressions"]):
        pass

    recommendation = comparison["recommendation"]
    lines.append("")
    lines.append("=" * 62)
    if recommendation["winner"]:
        lines.append(f"RECOMMENDED: {recommendation['winner']}")
    else:
        lines.append("RECOMMENDED: none")
    lines.append(f"  {recommendation['reason']}")
    if recommendation["disqualified"]:
        lines.append(
            f"  Disqualified on safety: {', '.join(recommendation['disqualified'])}"
        )
    lines.append("=" * 62)
    return "\n".join(lines)
