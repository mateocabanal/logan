from pathlib import Path

p = Path("tools/apply_issue82_recompile.py")
text = p.read_text()
old = ''''''    let (base_resident_bytes, memory_budget_bytes) = machine_base_reserve(model, target_profile, machine)?;\n    let input = OptimizeInput {\n        groups,\n        contexts: context_candidates(model, constraint),\n        context_constraint: constraint,\n        base_resident_bytes,\n        memory_budget_bytes,''''''
new = ''''''    let (base_resident_bytes, memory_budget_bytes) =\n        machine_base_reserve(model, target_profile, machine)?;\n    let input = OptimizeInput {\n        groups,\n        contexts: context_candidates(model, constraint),\n        context_constraint: constraint,\n        base_resident_bytes,\n        memory_budget_bytes,''''''
if old not in text:
    raise SystemExit("stale compiler optimizer anchor not found in recompile patch script")
p.write_text(text.replace(old, new, 1))
