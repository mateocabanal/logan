#!/usr/bin/env python3
"""Pinned-reference MiniCPM5 fixture oracle.

This is a development-only fixture producer.  It deliberately requires a
manifest containing a local model path, an upstream revision pin, and SHA-256
identity data before it imports the optional reference stack.  A successful
run writes deterministic NumPy arrays plus metadata describing the exact
source, geometry, numerical settings, inputs, and output paths.

The current reference adapter uses Hugging Face Transformers and PyTorch when
those optional packages are installed.  Logan does not import this module.

Manifest schema and an invocation example are in
``tools/minicpm5_oracle_manifest.json.example``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "logan.minicpm5.oracle-manifest"
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
MODEL_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


class OracleError(RuntimeError):
    """An input, identity, dependency, or reference-execution error."""


def _fail(message: str) -> None:
    raise OracleError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail(f"{label} is missing: {path}")
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail(f"cannot hash model file {path}: {exc}")
    return digest.hexdigest()


def _tree_sha256(root: Path, files: dict[str, str]) -> str:
    """Hash the ordered, individually pinned file set, including names."""
    digest = hashlib.sha256()
    for relative in sorted(files):
        path = root / relative
        data_hash = _sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(data_hash))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty path")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _validate_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        _fail(f"{label} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _validate_revision(value: Any) -> str:
    if not isinstance(value, str) or not REVISION_RE.fullmatch(value):
        _fail("model.revision must be an exact 40-character hexadecimal commit revision")
    return value.lower()


def _validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path, str, str, Path]:
    manifest = _read_json(manifest_path, "manifest")
    if manifest.get("schema") != SCHEMA:
        _fail(f"manifest schema must be {SCHEMA!r}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _fail(f"manifest schema_version must be {SCHEMA_VERSION}")

    model = manifest.get("model")
    if not isinstance(model, dict):
        _fail("manifest.model must be an object with path, revision, sha256, and files")
    model_root = _relative_path(manifest_path.parent, model.get("path"), "model.path")
    if not model_root.is_dir():
        _fail(f"model.path is not a directory: {model_root}")
    revision = _validate_revision(model.get("revision"))
    tree_sha = _validate_sha(model.get("sha256"), "model.sha256")
    pinned_files = model.get("files")
    if not isinstance(pinned_files, dict) or not pinned_files:
        _fail("model.files must be a non-empty object of relative paths to SHA-256 hashes")

    normalized_files: dict[str, str] = {}
    has_config = False
    has_weights = False
    for name, expected in pinned_files.items():
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            _fail(f"model.files contains an unsafe relative path: {name!r}")
        normalized_files[name] = _validate_sha(expected, f"model.files[{name!r}]")
        if name == "config.json":
            has_config = True
        if Path(name).suffix.lower() in MODEL_SUFFIXES:
            has_weights = True
    if not has_config:
        _fail("model.files must pin config.json")
    if not has_weights:
        _fail("model.files must pin at least one model weight file (.safetensors/.bin/etc.)")

    for name, expected in normalized_files.items():
        path = model_root / name
        if not path.is_file():
            _fail(f"pinned model file is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            _fail(f"pinned model file hash mismatch for {name}: expected {expected}, got {actual}")
    actual_tree = _tree_sha256(model_root, normalized_files)
    if actual_tree != tree_sha:
        _fail(f"model tree hash mismatch: expected {tree_sha}, got {actual_tree}")

    config = _read_json(model_root / "config.json", "model config")
    identity_candidates: list[tuple[str, str]] = []
    for filename in (".minicpm5_revision", "revision.txt"):
        identity_path = model_root / filename
        if identity_path.is_file():
            try:
                identity_candidates.append((filename, identity_path.read_text(encoding="utf-8").strip()))
            except OSError as exc:
                _fail(f"cannot read revision identity {identity_path}: {exc}")
    commit = config.get("_commit_hash")
    if isinstance(commit, str) and commit.strip():
        identity_candidates.append(("config.json:_commit_hash", commit.strip()))
    if not identity_candidates:
        _fail(
            "cannot validate pinned revision identity: model must contain "
            ".minicpm5_revision or revision.txt, or config.json:_commit_hash"
        )
    for source, candidate in identity_candidates:
        if candidate.lower() != revision:
            _fail(f"pinned revision mismatch in {source}: expected {revision}, got {candidate!r}")

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        _fail("manifest.tokenizer must contain path and sha256")
    tokenizer_path = _relative_path(model_root, tokenizer.get("path"), "tokenizer.path")
    if not tokenizer_path.is_file():
        _fail(f"tokenizer file is missing: {tokenizer_path}")
    tokenizer_sha = _validate_sha(tokenizer.get("sha256"), "tokenizer.sha256")
    actual_tokenizer_sha = _sha256_file(tokenizer_path)
    if actual_tokenizer_sha != tokenizer_sha:
        _fail(f"tokenizer hash mismatch: expected {tokenizer_sha}, got {actual_tokenizer_sha}")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        _fail("manifest.inputs must be an object")
    fixed = inputs.get("fixed_token_ids")
    if not isinstance(fixed, list) or not fixed or any(not isinstance(row, list) or not row for row in fixed):
        _fail("inputs.fixed_token_ids must be a non-empty list of non-empty token-id lists")
    for row_index, row in enumerate(fixed):
        for token_index, token in enumerate(row):
            if not isinstance(token, int) or isinstance(token, bool) or token < 0:
                _fail(f"inputs.fixed_token_ids[{row_index}][{token_index}] must be a non-negative integer")
    cached = inputs.get("cached_decode")
    if not isinstance(cached, dict):
        _fail("inputs.cached_decode must contain prompt_ids and decode_token_ids")
    for key in ("prompt_ids", "decode_token_ids"):
        row = cached.get(key)
        if not isinstance(row, list) or not row or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in row):
            _fail(f"inputs.cached_decode.{key} must be a non-empty list of non-negative integers")
    taps = inputs.get("target_taps", [])
    if not isinstance(taps, list) or any(not isinstance(tap, str) or not tap.strip() for tap in taps):
        _fail("inputs.target_taps must be a list of non-empty module paths")
    tap_names = [re.sub(r"[^A-Za-z0-9_.-]+", "_", tap).strip("_") or "tap" for tap in taps]
    if len(set(tap_names)) != len(tap_names):
        _fail("inputs.target_taps contains paths that collide after output-name sanitization")
    exports = manifest.get("exports", {})
    if not isinstance(exports, dict):
        _fail("manifest.exports must be an object")
    layer_indices = exports.get("one_layer_intermediates", [0])
    if not isinstance(layer_indices, list) or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in layer_indices
    ):
        _fail("exports.one_layer_intermediates must be a list of non-negative integers")

    numerical = manifest.get("numerical")
    if not isinstance(numerical, dict):
        _fail("manifest.numerical must be an object")
    if numerical.get("compute_dtype") not in {"float32", "float16", "bfloat16"}:
        _fail("numerical.compute_dtype must be float32, float16, or bfloat16")
    if numerical.get("output_dtype") not in {"float32", "float16"}:
        _fail("numerical.output_dtype must be float32 or float16")
    if numerical.get("device", "cpu") not in {"cpu", "cuda", "mps"}:
        _fail("numerical.device must be cpu, cuda, or mps")
    for key in ("atol", "rtol"):
        value = numerical.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            _fail(f"numerical.{key} must be a non-negative number")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        _fail("manifest.outputs must contain directory, arrays, and metadata paths")
    output_root = _relative_path(manifest_path.parent, outputs.get("directory"), "outputs.directory")
    output_paths: dict[str, Path] = {}
    for key in ("arrays", "metadata"):
        value = outputs.get(key)
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            _fail(f"outputs.{key} must be a safe relative path")
        output_paths[key] = (output_root / value).resolve()

    backend = manifest.get("backend", "transformers")
    for key, path in output_paths.items():
        if path.parent != output_root and output_root not in path.parents:
            _fail(f"outputs.{key} escapes outputs.directory")
    if output_paths["arrays"] == output_paths["metadata"]:
        _fail("outputs.arrays and outputs.metadata must be different files")

    backend = manifest.get("backend", "transformers")
    if backend != "transformers":
        _fail("manifest.backend must be 'transformers' (the optional Transformers/PyTorch reference adapter)")
    return manifest, model_root, revision, actual_tokenizer_sha, tokenizer_path


def _import_reference() -> tuple[Any, Any]:
    try:
        import numpy  # type: ignore
        import torch  # type: ignore
        import transformers  # type: ignore
    except ImportError as exc:
        _fail(
            "optional reference dependency is unavailable: install NumPy, PyTorch, "
            f"and Transformers to run the oracle ({exc})"
        )
    return torch, transformers


def _dtype(torch: Any, name: str) -> Any:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _first_tensor(value: Any, torch: Any) -> Any | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item, torch)
            if tensor is not None:
                return tensor
    return None


def _as_numpy(value: Any, torch: Any, output_dtype: str) -> Any:
    # NumPy has no universally portable bfloat16, so convert only at the
    # serialization boundary; the reference forward still uses compute_dtype.
    tensor = value.detach().to(dtype=_dtype(torch, output_dtype)).cpu()
    return tensor.numpy()


def _module_by_path(model: Any, path: str) -> Any:
    current = model
    for component in path.split("."):
        if not component:
            _fail(f"target tap has an empty module path component: {path!r}")
        if component.isdigit() and hasattr(current, "__getitem__"):
            try:
                current = current[int(component)]
                continue
            except (IndexError, KeyError, TypeError):
                pass
        if not hasattr(current, component):
            _fail(f"configured target tap module does not exist: {path}")
        current = getattr(current, component)
    return current


class TransformersReference:
    def __init__(self, model_root: Path, tokenizer_path: Path, manifest: dict[str, Any], torch: Any, transformers: Any):
        self.torch = torch
        self.transformers = transformers
        numerical = manifest["numerical"]
        device_name = numerical.get("device", "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            _fail("numerical.device=cuda but PyTorch reports no CUDA device")
        if device_name == "mps":
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is None or not mps.is_available():
                _fail("numerical.device=mps but PyTorch reports no MPS device")
        self.device = torch.device(device_name)
        self.output_dtype = numerical["output_dtype"]
        compute_dtype = _dtype(torch, numerical["compute_dtype"])
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(tokenizer_path.parent), local_files_only=True,
                trust_remote_code=bool(manifest["model"].get("trust_remote_code", False)),
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                str(model_root), local_files_only=True, torch_dtype=compute_dtype,
                trust_remote_code=bool(manifest["model"].get("trust_remote_code", False)),
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            _fail(f"reference dependency could not load the pinned MiniCPM5 model locally: {exc}")

    def _tensor_ids(self, tokens: list[int]) -> Any:
        ids = self.torch.tensor([tokens], dtype=self.torch.long, device=self.device)
        vocab = getattr(self.model.config, "vocab_size", None)
        if vocab is not None and any(token >= int(vocab) for token in tokens):
            _fail(f"token id exceeds model vocabulary ({vocab})")
        return ids

    def _forward(self, ids: Any, *, cache: Any = None, hidden: bool = True) -> Any:
        kwargs: dict[str, Any] = {
            "input_ids": ids,
            "use_cache": cache is not None or hidden,
            "output_hidden_states": hidden,
            "return_dict": True,
        }
        if cache is not None:
            kwargs["past_key_values"] = cache
        try:
            with self.torch.no_grad():
                return self.model(**kwargs)
        except Exception as exc:
            _fail(f"reference forward failed for MiniCPM5: {exc}")

    def run(self, manifest: dict[str, Any]) -> dict[str, Any]:
        inputs = manifest["inputs"]
        output_dtype = self.output_dtype
        arrays: dict[str, Any] = {}
        taps: dict[str, Any] = {}
        handles: list[Any] = []
        for tap_path in inputs.get("target_taps", []):
            module = _module_by_path(self.model, tap_path)

            def capture(_module: Any, _args: Any, output: Any, name: str = tap_path) -> None:
                tensor = _first_tensor(output, self.torch)
                if tensor is not None:
                    taps[name] = _as_numpy(tensor, self.torch, output_dtype)

            handles.append(module.register_forward_hook(capture))
        try:
            fixed_rows = inputs["fixed_token_ids"]
            layer_indices = manifest.get("exports", {}).get("one_layer_intermediates", [0])
            for row_index, tokens in enumerate(fixed_rows):
                taps.clear()
                result = self._forward(self._tensor_ids(tokens), hidden=True)
                hidden_states = getattr(result, "hidden_states", None)
                if not hidden_states:
                    _fail("reference model did not return hidden states; embedding/intermediate exports are unavailable")
                embeddings = hidden_states[0]
                arrays[f"fixed_{row_index:03d}_embeddings"] = _as_numpy(embeddings, self.torch, output_dtype)[0]
                for layer_index in layer_indices:
                    if layer_index + 1 >= len(hidden_states):
                        _fail(f"one_layer_intermediates index {layer_index!r} is not present in model hidden states")
                    arrays[f"fixed_{row_index:03d}_layer_{layer_index:03d}"] = _as_numpy(hidden_states[layer_index + 1], self.torch, output_dtype)[0]
                arrays[f"fixed_{row_index:03d}_final_logits"] = _as_numpy(result.logits, self.torch, output_dtype)[0]
                for name, value in sorted(taps.items()):
                    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "tap"
                    arrays[f"fixed_{row_index:03d}_tap_{safe}"] = value

            cached = inputs["cached_decode"]
            prompt = cached["prompt_ids"]
            decode_tokens = cached["decode_token_ids"]
            prompt_result = self._forward(self._tensor_ids(prompt), hidden=False)
            past = getattr(prompt_result, "past_key_values", None)
            if past is None:
                _fail("reference model did not return past_key_values; cached decode cannot be exported")
            decode_logits: list[Any] = []
            for token in decode_tokens:
                step = self._forward(self._tensor_ids([token]), cache=past, hidden=False)
                past = getattr(step, "past_key_values", None)
                if past is None:
                    _fail("reference model dropped past_key_values during cached decode")
                decode_logits.append(_as_numpy(step.logits, self.torch, output_dtype)[0, 0])
            arrays["cached_decode_logits"] = self._stack(decode_logits)
            arrays["cached_decode_token_ids"] = self._numpy_int64(decode_tokens)
        finally:
            for handle in handles:
                handle.remove()
        return arrays

    def _stack(self, values: list[Any]) -> Any:
        if not values:
            _fail("cached decode produced no logits")
        return self.__class__._np_stack(values)

    @staticmethod
    def _np_stack(values: list[Any]) -> Any:
        import numpy as np
        return np.stack(values, axis=0)

    @staticmethod
    def _numpy_int64(values: list[int]) -> Any:
        import numpy as np
        return np.asarray(values, dtype=np.int64)


def _serialize_arrays(arrays: dict[str, Any]) -> bytes:
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **{key: arrays[key] for key in sorted(arrays)})
    return stream.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Produce pinned MiniCPM5 numerical fixtures using the optional "
            "Transformers/PyTorch reference stack. No runtime code depends on this tool."
        ),
        epilog=(
            "The manifest must pin model.path, a 40-hex revision, every model-file "
            "SHA-256, a tree SHA-256, tokenizer SHA-256, fixed token IDs, and output paths. "
            "Use tools/minicpm5_oracle_manifest.json.example as the schema template."
        ),
    )
    parser.add_argument("manifest", type=Path, help="deterministic oracle manifest JSON")
    args = parser.parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        manifest, model_root, revision, tokenizer_sha, tokenizer_path = _validate_manifest(manifest_path)
        torch, transformers = _import_reference()
        reference = TransformersReference(model_root, tokenizer_path, manifest, torch, transformers)
        arrays = reference.run(manifest)
        array_bytes = _serialize_arrays(arrays)

        outputs = manifest["outputs"]
        output_root = _relative_path(manifest_path.parent, outputs["directory"], "outputs.directory")
        array_path = (output_root / outputs["arrays"]).resolve()
        metadata_path = (output_root / outputs["metadata"]).resolve()
        config = _read_json(model_root / "config.json", "model config")
        geometry_keys = (
            "model_type",
            "architectures",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "rope_parameters",
            "bos_token_id",
            "eos_token_id",
            "tie_word_embeddings",
        )
        metadata = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "kind": "development_oracle_fixture",
            "source": {
                "model_path": str(model_root),
                "revision": revision,
                "model_sha256": manifest["model"]["sha256"].lower(),
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_sha256": tokenizer_sha,
                "manifest_sha256": hashlib.sha256(_canonical_json(manifest)).hexdigest(),
            },
            "geometry": {key: config[key] for key in geometry_keys if key in config},
            "dtype": {
                "source_torch_dtype": config.get("torch_dtype"),
                "compute_dtype": manifest["numerical"]["compute_dtype"],
                "output_dtype": manifest["numerical"]["output_dtype"],
            },
            "numerical": manifest["numerical"],
            "inputs": manifest["inputs"],
            "exports": manifest.get("exports", {"one_layer_intermediates": [0]}),
            "arrays": {name: _shape_dtype(value) for name, value in sorted(arrays.items())},
            "output_paths": {"arrays": str(array_path), "metadata": str(metadata_path)},
            "arrays_sha256": hashlib.sha256(array_bytes).hexdigest(),
            "claims": [],
        }
        _write_atomic(array_path, array_bytes)
        _write_atomic(metadata_path, (_canonical_json(metadata) + b"\n"))
        print(f"wrote {array_path}")
        print(f"wrote {metadata_path}")
        return 0
    except OracleError as exc:
        print(f"minicpm5-oracle: error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"minicpm5-oracle: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
