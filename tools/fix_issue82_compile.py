from pathlib import Path

p = Path("logan-compiler/src/pipeline.rs")
text = p.read_text()
patterns = [
    ("""        let exact = record_inventory(\n            &model,\n            ExpertQuantization::Exact,\n            target::LINUX_X86_64_AVX2_V1,\n        )""",
     """        let exact = record_inventory(\n            &model,\n            ExpertQuantization::Exact,\n            None,\n            target::LINUX_X86_64_AVX2_V1,\n        )"""),
    ("""        let mxfp4 = record_inventory(\n            &model,\n            ExpertQuantization::Mxfp4,\n            target::LINUX_X86_64_AVX2_V1,\n        )""",
     """        let mxfp4 = record_inventory(\n            &model,\n            ExpertQuantization::Mxfp4,\n            None,\n            target::LINUX_X86_64_AVX2_V1,\n        )"""),
    ("""        let records = record_inventory(\n            &model,\n            ExpertQuantization::Mxfp4,\n            target::MACOS_ARM64_METAL_APPLE8_V1,\n        )""",
     """        let records = record_inventory(\n            &model,\n            ExpertQuantization::Mxfp4,\n            None,\n            target::MACOS_ARM64_METAL_APPLE8_V1,\n        )"""),
]
for old, new in patterns:
    if old not in text:
        raise SystemExit(f"expected call site not found: {old!r}")
    text = text.replace(old, new, 1)
p.write_text(text)
