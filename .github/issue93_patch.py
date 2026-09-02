from pathlib import Path

path = Path('logan-compiler/src/pipeline.rs')
text = path.read_text()
old = '''    let memory = MemoryPlan {
        placement,
        quant,
        layout,
        ram_budget_bytes: pool,
    };'''
new = '''    let memory = MemoryPlan {
        placement,
        quant,
        layout,
        // #93 makes ResourcePlan authoritative. Existing compiler lowering is
        // migrated incrementally; until a value has an explicit resource
        // decision, legacy placement remains the compatibility projection.
        resources: Vec::new(),
        ram_budget_bytes: pool,
    };'''
if old not in text:
    raise SystemExit('expected MemoryPlan constructor not found')
path.write_text(text.replace(old, new, 1))
