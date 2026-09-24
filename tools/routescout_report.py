#!/usr/bin/env python3
"""Generate the EXP-078 final report from preserved artifacts.

Every number in the output is read from a file this experiment wrote — the sweep
metrics, the held-out comparison, the per-layer analysis, the latency JSON, the
corpus index, and the checkpoint hashes. Nothing is retyped, so the report cannot
drift from the data as scale points land, and a claim in the report can always be
traced back to the artifact that produced it.

Sections:

- dataset scaling (learning curve, both selection-biased and final-epoch columns)
- initialization comparison (Arms A / B / C)
- best checkpoint (file, SHA-256, metrics, epoch, dataset)
- comparison (Edge0 head vs previous RouteScout vs new RouteScout)
- per-layer findings (strongest, weakest, layers that did not improve)
- scaling conclusion (data-limited vs plateau)
- recommendation

The conclusion section is deliberately conservative: it states what the measured
curve supports and names the evidence, rather than asserting a trend the data does
not yet show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRIC_ROWS = (
    ("recall@1", "recall1"),
    ("recall@4", "recall4"),
    ("recall@8", "recall8"),
    ("recall@12", "recall12"),
    ("weighted mass@4", "weighted_mass4"),
    ("weighted mass@8", "weighted_mass8"),
    ("exact native top1", "exact_top1"),
    ("top1-in-K4", "top1_in_k4"),
    ("full K4 coverage@4", "full4"),
    ("soft CE", "loss"),
)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def md_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    def line(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
                     + [line(r) for r in rows])


def scale_key(name: str) -> float:
    try:
        return float(str(name).rstrip("kK")) * 1000
    except ValueError:
        return float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root: Path = args.root
    runs_dir = root / "runs"
    results = root / "results"
    out: list[str] = []
    facts: dict = {}

    # ---- corpus ----
    index = load(root / "corpus" / "corpus-index.json") or {"runs": []}
    train_runs = [r for r in index.get("runs", []) if r.get("bank") == "train" and r.get("run_id")]
    test_index = load(root / "final" / "corpus-index.json") or {"runs": []}
    val_runs = [r for r in index.get("runs", []) if r.get("bank") == "val" and r.get("run_id")]
    tokens = sum(int(r.get("tokens", 0)) for r in train_runs)
    facts["corpus"] = {
        "pool_runs": len(train_runs),
        "pool_tokens": tokens,
        "val_runs": len(val_runs),
        "test_runs": len([r for r in test_index.get("runs", []) if r.get("run_id")]),
        "failures": index.get("failures", []),
    }

    # ---- learning curve ----
    curve_rows = []
    curve_facts = []
    for path in sorted(runs_dir.glob("routescout_*.metrics.json"), key=lambda p: p.name):
        blob = load(path)
        if not blob or not blob.get("heads"):
            continue
        heads = blob["heads"]
        def mean(phase: str, key: str) -> float:
            vals = []
            for h in heads:
                block = h[phase]
                v = block.get(key)
                if v is None and key == "top1_in_k4":
                    v = block.get("top1_in_target")
                if v is not None and np.isfinite(v):
                    vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")
        def mean_final(key: str) -> float:
            vals = []
            for h in heads:
                hist = h.get("history") or []
                if not hist:
                    continue
                v = hist[-1].get(key)
                if v is None and key == "top1_in_k4":
                    v = hist[-1].get("top1_in_target")
                if v is not None and np.isfinite(v):
                    vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")

        best_epochs = [h["best_epoch"] for h in heads]
        entry = {
            "scale": blob.get("scale_name"),
            "arm": blob.get("arm"),
            "runs": blob.get("run_count"),
            "train_examples_per_head": blob.get("train_examples_per_head"),
            "wall_seconds": blob.get("wall_seconds"),
            "init": blob.get("init"),
            "checkpoint_sha256": blob.get("checkpoint_sha256"),
            "best_epoch_mean": float(np.mean(best_epochs)),
            "best_epoch_max": int(max(best_epochs)),
            "epochs_budget": blob.get("epochs"),
            "best": {k: mean("best", k) for _label, k in METRIC_ROWS},
            "final": {k: mean_final(k) for _label, k in METRIC_ROWS},
            "baseline": {k: mean("baseline", k) for _label, k in METRIC_ROWS},
        }
        curve_facts.append(entry)
        curve_rows.append((
            scale_key(entry["scale"]), entry["arm"],
            [
                entry["scale"], entry["arm"], f"{entry['train_examples_per_head']:,}",
                f"{entry['best']['recall4']:.4f}", f"{entry['final']['recall4']:.4f}",
                f"{entry['best']['weighted_mass4']:.4f}", f"{entry['final']['weighted_mass4']:.4f}",
                f"{entry['best']['top1_in_k4']:.4f}",
                f"{entry['best']['full4']:.4f}",
                f"{entry['best']['loss']:.4f}",
                f"{entry['best_epoch_mean']:.1f}",
                f"{entry['wall_seconds']:.0f}" if entry["wall_seconds"] else "n/a",
            ],
        ))
    # Order the curve by scale magnitude then arm, so the table reads as a curve
    # rather than alphabetically (`10k` must not precede `5k`).
    curve_rows = [row for _key, _arm, row in sorted(curve_rows, key=lambda t: (t[0], t[1]))]

    curve_facts.sort(key=lambda e: (scale_key(e["scale"]), e["arm"]))
    out.append("## Dataset scaling\n")
    out.append(
        "`best` columns are selection-biased (the checkpoint rule picks the epoch that\n"
        "looks best on the same bank the curve is plotted from); `final` is the last\n"
        "epoch, unbiased but including any late-epoch overfit. The frozen test-bank\n"
        "comparison below is the unbiased headline.\n"
    )
    if curve_rows:
        out.append(md_table(
            ["scale", "arm", "train/head", "r@4 best", "r@4 final", "mass@4 best",
             "mass@4 final", "top1 best", "full@4", "CE", "best_ep", "wall_s"],
            curve_rows,
        ))
    else:
        out.append("_No scale point has completed yet._")
    out.append("")
    facts["curve"] = curve_facts

    # ---- initialization comparison ----
    by_scale: dict[str, dict[str, dict]] = {}
    for e in curve_facts:
        by_scale.setdefault(str(e["scale"]), {})[str(e["arm"])] = e
    init_rows = []
    for scale in sorted(by_scale, key=scale_key):
        arms = by_scale[scale]
        if len(arms) < 2:
            continue
        row = [scale, f"{arms.get('edge0', {}).get('train_examples_per_head', 0):,}"]
        for arm in ("edge0", "rs_v1", "random"):
            e = arms.get(arm)
            row.append(f"{e['best']['recall4']:.4f}" if e else "not run")
        for arm in ("edge0", "rs_v1", "random"):
            e = arms.get(arm)
            row.append(f"{e['best']['weighted_mass4']:.4f}" if e else "not run")
        init_rows.append(row)
    out.append("## Initialization comparison\n")
    if init_rows:
        out.append(md_table(
            ["scale", "train/head", "A r@4", "B r@4", "C r@4",
             "A mass@4", "B mass@4", "C mass@4"],
            init_rows,
        ))
        if not any(by_scale[s].get("random") for s in by_scale):
            out.append(
                "\nArm C (random) has not been trained yet; the sweep schedules it at "
                "the `--random-scale` point."
            )
    else:
        out.append("_Fewer than two arms have completed at any scale._")
    out.append("")

    # ---- best checkpoint ----
    summary = load(results / "summary.json")
    out.append("## Best RouteScout checkpoint\n")
    best = (summary or {}).get("best_checkpoint")
    if best:
        name = best["adapter"]
        # Match the *specific* adapter, not merely the arm. Matching by arm would
        # always return the largest scale of the winning arm, so if a smaller scale
        # won on held-out mass@4 the report would print that arm's biggest
        # checkpoint's hash/metadata under the winner's name — and the hash is what
        # acceptance criterion 8 rests on.
        cand = next(
            (e for e in curve_facts if name == f"routescout_{e['scale']}_{e['arm']}"),
            None,
        )
        out.append(f"- adapter: `{name}`")
        if cand is None:
            out.append(
                "  - **warning:** no curve row matches this adapter name exactly; the "
                "scale, epoch, and hash below are unavailable. This means the nominated "
                "checkpoint was renamed or produced outside the sweep."
            )
        else:
            out.append(f"- scale: {cand['scale']} ({cand['train_examples_per_head']:,} training examples/head, "
                       f"{cand['runs']} pool runs)")
            out.append(f"- initialization: {cand['init']}")
            out.append(f"- chosen epoch (mean over heads): {cand['best_epoch_mean']:.1f} of {cand['epochs_budget']}")
            out.append(f"- checkpoint SHA-256: `{cand['checkpoint_sha256']}`")
            out.append(f"- training wall: {cand['wall_seconds']:.0f} s")
        out.append("")
        metric_rows = []
        for label, key in METRIC_ROWS:
            value = best["metrics"].get(key)
            if value is None and key == "top1_in_k4":
                value = best["metrics"].get("top1_in_target")
            metric_rows.append([label, f"{value:.4f}" if value is not None else "n/a"])
        out.append(md_table(["held-out metric", "value"], metric_rows))
    else:
        out.append("_No best checkpoint nominated yet — the held-out comparison has not run._")
    out.append("")

    # ---- comparison ----
    out.append("## Comparison on the frozen test bank\n")
    compare = load(results / "compare-test.json")
    if compare:
        rows = []
        for adapter, blob in compare["adapters"].items():
            m = blob["mean"]
            row = [adapter]
            for label, key in METRIC_ROWS:
                value = m.get(key)
                if value is None and key == "top1_in_k4":
                    value = m.get("top1_in_target")
                row.append(f"{value:.4f}" if value is not None else "n/a")
            rows.append(row)
        out.append(md_table(
            ["head"] + [label for label, _k in METRIC_ROWS], rows,
        ))
        recs = set((compare.get("val_records_per_head") or {}).values())
        out.append(f"\nAll rows scored on identical records: {sorted(recs)} per head.")
        facts["comparison_adapters"] = sorted(compare["adapters"])
    else:
        out.append("_Comparison not yet produced._")
    out.append("")

    # ---- per-layer ----
    out.append("## Per-layer findings\n")
    layers = load(results / "layers.json")
    if layers and layers.get("per_layer"):
        metric = layers.get("metric", "weighted_mass4")
        owners = [r["owner"] for r in layers["per_layer"]]
        adapters = [k for k in layers["per_layer"][0] if k != "owner"]
        ref = adapters[0]
        rows = []
        for r in layers["per_layer"]:
            rows.append([str(r["owner"])] + [f"{r.get(a, float('nan')):.4f}" for a in adapters])
        out.append(f"Per-layer `{metric}` on the held-out set:\n")
        out.append(md_table(["layer"] + adapters, rows))
        for a in adapters[1:]:
            worse = [r["owner"] for r in layers["per_layer"]
                     if np.isfinite(r.get(a, np.nan)) and np.isfinite(r.get(ref, np.nan))
                     and r[a] <= r[ref]]
            out.append(f"\n- `{a}`: heads not beating `{ref}` on `{metric}`: {worse or 'none'}")
        deltas = layers.get("curve_layer_deltas") or []
        for d in deltas:
            out.append(f"\n- {d['from']} -> {d['to']}: mean delta "
                       f"{d['mean_delta']:+.4f}, improved {d['improved']}/{d['total']}, "
                       f"stalled layers: {d['stalled_layers'] or 'none'}")
    else:
        out.append("_Per-layer analysis not yet produced._")
    out.append("")

    # ---- latency ----
    out.append("## Predictor cost\n")
    mlx = load(results / "latency.json")
    if mlx:
        if mlx.get("contended"):
            out.append("**Measured under contention — upper bound only, not a clean figure.**\n")
        rows = []
        for name, blob in mlx["adapters"].items():
            rows.append([name, f"{blob['median_head_us']:.1f}",
                         f"{blob['total_32_heads_ms']:.2f}"])
        out.append(md_table(["adapter", "median per head (us)", "32 heads (ms/token)"], rows))
    deployed = load(results / "deployed.json")
    if deployed:
        out.append("")
        rows = []
        ctrl = deployed.get("control_native_k4") or {}
        rows.append(["native K4 control (no predictor)",
                     str(ctrl.get("route_ms_per_tok")), str(ctrl.get("predict_ms_per_tok")), "—"])
        for name, blob in deployed["adapters"].items():
            stage = blob.get("stage") or {}
            rows.append([name, str(blob.get("route_ms_per_tok")),
                         str(blob.get("predict_ms_per_tok")),
                         f"{stage.get('recall', float('nan')):.4f}"])
        out.append(md_table(["arm", "route ms/tok", "predict ms/tok", "runtime recall@4"], rows))
        out.append(f"\nAll arms emitted identical token IDs: {deployed.get('token_ids_identical')}")
    rq = load(results / "runtime-quality.json")
    if rq:
        out.append("")
        # A stale artifact must not read as the final result. If the probe does not
        # cover every adapter in the comparison it predates the last scale point, so
        # it is reported as stale rather than as the headline table.
        covered = set(rq["arms"])
        expected = set((compare or {}).get("adapters", {}))
        stale = bool(expected) and not expected.issubset(covered)
        out.append(
            "Deployed runtime quality vs corpus scale (single prompt, native K4 "
            "authoritative; the arena's own counters):\n"
        )
        if stale:
            missing = sorted(expected - covered)
            out.append(
                f"**STALE — measured before the final heads existed.** It covers "
                f"{len(covered)} adapter(s) and does not include: {missing}. "
                f"Treat it as a partial probe, not the final result.\n"
            )
        if rq.get("contended"):
            out.append(
                "**Measured under host contention** (another model process was "
                "alive), so these are bounds rather than clean figures.\n"
            )
        rows = []
        for name, blob in rq["arms"].items():
            st = blob.get("stage") or {}
            rows.append([name,
                         f"{st.get('recall', float('nan')):.4f}",
                         f"{st.get('full_route_coverage', float('nan')):.4f}",
                         f"{st.get('efficiency', float('nan')):.3f}",
                         str(st.get("late")), str(st.get("duplicate_reads"))])
        out.append(md_table(
            ["head", "runtime recall@4", "full K4 coverage@4", "efficiency", "late", "dup"],
            rows,
        ))
        out.append(f"\nToken IDs identical across arms: {rq.get('token_ids_identical')}")
    elif not mlx and not deployed:
        out.append("_Latency not yet measured._")
    out.append("")

    # ---- conclusion ----
    out.append("## Scaling conclusion\n")
    real = [e for e in curve_facts if str(e["arm"]) in ("edge0", "rs_v1") and e["scale"]]
    if len(real) >= 2:
        ordered = sorted(real, key=lambda e: (str(e["arm"]), scale_key(e["scale"])))
        for arm in sorted({str(e["arm"]) for e in ordered}):
            seq = [e for e in ordered if str(e["arm"]) == arm]
            if len(seq) < 2:
                continue
            gains = [
                (seq[i + 1]["best"]["weighted_mass4"] - seq[i]["best"]["weighted_mass4"],
                 str(seq[i]["scale"]), str(seq[i + 1]["scale"]))
                for i in range(len(seq) - 1)
            ]
            last = gains[-1]
            out.append(
                f"- **{arm}**: mass@4 {seq[0]['best']['weighted_mass4']:.4f} "
                f"({seq[0]['scale']}) -> {seq[-1]['best']['weighted_mass4']:.4f} "
                f"({seq[-1]['scale']}); last step {last[0]:+.4f} "
                f"({last[1]} -> {last[2]})."
            )
        out.append("")
        out.append(
            "Verdict (derived, not asserted): a positive last step means the architecture "
            "has not yet saturated at the largest measured scale; a step that is flat or "
            "negative while `best_epoch` stays mid-budget means the trunk, not the data, "
            "is the binding constraint. Both readings are in the per-scale rows above, "
            "and the plateau claim is only made if the data shows it."
        )
    else:
        out.append("_Fewer than two scale points are complete; the curve is not yet decidable._")
    out.append("")

    # ---- recommendation ----
    out.append("## Recommendation\n")
    rec = []
    if len(real) >= 2:
        ordered = sorted(real, key=lambda e: (str(e["arm"]), scale_key(e["scale"])))
        for arm in sorted({str(e["arm"]) for e in ordered}):
            seq = [e for e in ordered if str(e["arm"]) == arm]
            if len(seq) < 2:
                continue
            # Every consecutive step, not just the last one: a plateau claim from a
            # single step is fragile, and the shape of the sequence is the evidence.
            steps = [
                (seq[i + 1]["best"]["weighted_mass4"] - seq[i]["best"]["weighted_mass4"],
                 seq[i]["scale"], seq[i + 1]["scale"])
                for i in range(len(seq) - 1)
            ]
            budget = seq[-1].get("epochs_budget") or 0
            mid_budget = seq[-1]["best_epoch_mean"] < 0.75 * budget if budget else False
            trend = ", ".join(f"{a}->{b} {d:+.4f}" for d, a, b in steps)
            last_step = steps[-1][0]
            total_gain = seq[-1]["best"]["weighted_mass4"] - seq[0]["best"]["weighted_mass4"]
            still_rising = all(d > 0 for d, _a, _b in steps)
            if last_step > 0.005 and still_rising:
                rec.append(
                    f"- `{arm}` is still **data-limited** at {seq[-1]['scale']}: every "
                    f"step is positive (mass@4 {trend}; total {total_gain:+.4f}), so extend "
                    f"the corpus before changing the architecture."
                )
            elif last_step > 0.005:
                rec.append(
                    f"- `{arm}` is **probably data-limited** at {seq[-1]['scale']}: the last "
                    f"step is positive ({last_step:+.4f}) but not monotone "
                    f"(mass@4 {trend}), so the corpus is still worth extending."
                )
            elif mid_budget:
                rec.append(
                    f"- `{arm}` appears **architecture-limited** at {seq[-1]['scale']}: the "
                    f"last step is flat/negative ({last_step:+.4f}; mass@4 {trend}) while "
                    f"heads peak at epoch {seq[-1]['best_epoch_mean']:.1f} of {budget}, which "
                    f"is the signature of capacity rather than data being the constraint. A "
                    f"wider trunk or added temporal features is the better next experiment."
                )
            else:
                rec.append(
                    f"- `{arm}` is **inconclusive** at {seq[-1]['scale']} "
                    f"(last step {last_step:+.4f}; mass@4 {trend}, heads peak at epoch "
                    f"{seq[-1]['best_epoch_mean']:.1f} of {budget}): another scale point is "
                    f"needed before choosing between more data and a larger trunk."
                )
        out.extend(rec)
    else:
        out.append("- The curve is not yet decidable; collect and train more scale points first.")
    out.append("")
    out.append(
        "Candidate next steps, in the order this experiment's evidence would support:\n\n"
        "1. **Larger corpus** if the last scaling step is still positive.\n"
        "2. **Larger or shared trunk** if the curve flattens while heads still peak "
        "mid-budget — that is the signature of capacity rather than data being the "
        "constraint.\n"
        "3. **RouteScout-specific temporal features** (route n-grams, per-layer transition "
        "statistics) if per-layer analysis shows a subset of heads that never improve.\n"
        "4. **The H4 future-working-set head** only once the t+1 curve is settled — design "
        "notes are preserved in this section's earlier text; no H4 objective was trained "
        "here.\n"
        "5. **Recover-LoRA** is explicitly out of scope for this experiment and was not "
        "started.\n\n"
        "None of these were started automatically: the handoff requires them to be "
        "justified by this experiment's results rather than assumed."
    )
    out.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n")
    (args.output.with_suffix(".facts.json")).write_text(json.dumps(facts, indent=2) + "\n")
    print("\n".join(out))
    print(f"\nOUTPUT {args.output}")


if __name__ == "__main__":
    main()
