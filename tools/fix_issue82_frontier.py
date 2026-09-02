from pathlib import Path

p = Path("logan-ir/src/optimizer.rs")
text = p.read_text()
old = '''    let representatives = representative_indices(&non_dominated);\n    let mut result = Vec::with_capacity(representatives.len());\n    for (index, label) in representatives {\n        let mut plan = non_dominated[index].clone();\n        plan.label = Some(label.to_string());\n        result.push(plan);\n    }\n    Ok(result)\n'''
new = '''    let representatives = representative_indices(&non_dominated);\n    for (index, label) in representatives {\n        non_dominated[index].label = Some(label.to_string());\n    }\n    Ok(non_dominated)\n'''
if old not in text:
    raise SystemExit("expected Pareto representative return block not found")
p.write_text(text.replace(old, new, 1))
