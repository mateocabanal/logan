from pathlib import Path

p = Path('logan-compiler/src/recompile.rs')
s = p.read_text()
s = s.replace(
    'fn source_rans_table(package: &Package) -> Result<Option<RansTable>> {',
    'fn source_rans_table(package: &Package) -> Result<Option<rans256::Table>> {',
)
s = s.replace(
    '    rans_table: Option<&RansTable>,',
    '    rans_table: Option<&rans256::Table>,',
)
old = '''fn write_json_synced(path: &Path, value: &serde_json::Value) -> Result<()> {\n    let bytes = serde_json::to_vec_pretty(value)\n        .map_err(|error| ColicError::Usage(format!("cannot encode recompile state: {error}")))?;\n    write_synced(path, &bytes)\n}\n'''
new = '''fn write_json_synced(path: &Path, value: &serde_json::Value) -> Result<()> {\n    let bytes = serde_json::to_vec_pretty(value)\n        .map_err(|error| ColicError::Usage(format!("cannot encode recompile state: {error}")))?;\n    let name = path\n        .file_name()\n        .and_then(|name| name.to_str())\n        .ok_or_else(|| ColicError::Usage("recompile state path has no file name".into()))?;\n    let next = path.with_file_name(format!("{name}.next"));\n    write_synced(&next, &bytes)?;\n    #[cfg(not(windows))]\n    fs::rename(&next, path).map_err(|source| ColicError::Io {\n        path: path.to_owned(),\n        source,\n    })?;\n    #[cfg(windows)]\n    {\n        fs::copy(&next, path).map_err(|source| ColicError::Io {\n            path: path.to_owned(),\n            source,\n        })?;\n        File::open(path)\n            .and_then(|file| file.sync_all())\n            .map_err(|source| ColicError::Io {\n                path: path.to_owned(),\n                source,\n            })?;\n        remove_if_exists(&next)?;\n    }\n    Ok(())\n}\n'''
if old not in s:
    raise SystemExit('write_json_synced anchor missing')
s = s.replace(old, new, 1)
p.write_text(s)
Path('tools/zz_fix_lowspace_journal.py').unlink(missing_ok=True)
