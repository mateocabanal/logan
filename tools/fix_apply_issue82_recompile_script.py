from pathlib import Path

p = Path("tools/apply_issue82_recompile.py")
text = p.read_text()

# Replace the compiler-optimizer section with formatting-insensitive edits.
start_marker = "# Compiler optimizer uses dense/static bytes as the constant package component.\n"
end_marker = "# Share the interactive/noninteractive plan selector with recompile.\n"
start = text.find(start_marker)
end = text.find(end_marker)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("compiler optimizer patch section markers not found")
replacement = '''# Compiler optimizer uses dense/static bytes as the constant package component.
p = Path("logan-compiler/src/optimizer.rs")
source = p.read_text()
call = "machine_base_reserve(model, target_profile, machine)?;\\n"
pos = source.find(call)
if pos < 0:
    raise SystemExit("machine_base_reserve call not found in compiler optimizer")
insert_at = pos + len(call)
source = source[:insert_at] + "    let base_package_bytes = dense_resident_bytes(model)?;\\n" + source[insert_at:]
fields = "        base_resident_bytes,\\n        memory_budget_bytes,"
if fields not in source:
    raise SystemExit("OptimizeInput base fields not found in compiler optimizer")
source = source.replace(fields, "        base_resident_bytes,\\n        base_package_bytes,\\n        memory_budget_bytes,", 1)
p.write_text(source)

'''
text = text[:start] + replacement + text[end:]

# Replace the selector-sharing section without depending on rustfmt wrapping.
start_marker = "# Share the interactive/noninteractive plan selector with recompile.\n"
end_marker = "# Recompile CLI: selected plan + explicit quant intent.\n"
start = text.find(start_marker)
end = text.find(end_marker)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("selector patch section markers not found")
replacement = '''# Share the interactive/noninteractive plan selector with recompile.
p = Path("logan-compiler/src/pipeline.rs")
source = p.read_text()
needle = "fn choose_optimizer_plan(\\n"
if needle not in source:
    raise SystemExit("choose_optimizer_plan function not found in pipeline")
source = source.replace(needle, "pub(crate) fn choose_optimizer_plan(\\n", 1)
p.write_text(source)

'''
text = text[:start] + replacement + text[end:]
p.write_text(text)
