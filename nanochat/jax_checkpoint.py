"""Self-contained, versioned serving checkpoints for the JAX NanoChat model.

The directory contract is intentionally small and strict::

    manifest.json
    variables.msgpack
    tokenizer/tokenizer.json

``manifest.json`` is written last and therefore acts as the publication marker.
There is no legacy JAX checkpoint format in this project; incompatible schema or
architecture changes must use a new schema version instead of guessing defaults.
"""

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.typing import VariableDict

from nanochat.common_jax import GPTConfig
from nanochat.gpt_jax import GPT
from nanochat.tokenizer import HuggingFaceTokenizer

CHECKPOINT_SCHEMA = "mgr.nanochat.jax-serving.v1"
CHECKPOINT_ARCHITECTURE = "nanochat.gpt_jax.GPT"
VARIABLES_FORMAT = "flax.serialization.msgpack"
TOKENIZER_FORMAT = "huggingface-tokenizers-json"
MANIFEST_FILENAME = "manifest.json"
VARIABLES_FILENAME = "variables.msgpack"
TOKENIZER_DIRECTORY = "tokenizer"
TOKENIZER_FILENAME = "tokenizer.json"


class JaxCheckpointError(RuntimeError):
    """A serving checkpoint is missing, incompatible, or corrupt."""


@dataclass(frozen=True, slots=True)
class JaxServingCheckpoint:
    """Validated model state ready for publication by the inference server."""

    checkpoint_dir: Path
    step: int
    model: GPT
    variables: VariableDict
    tokenizer: HuggingFaceTokenizer
    config: GPTConfig


def _require_exact_keys(data: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(data)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise JaxCheckpointError(
        f"{label} keys do not match the checkpoint contract: missing={missing}, unexpected={unexpected}"
    )


def _require_int(data: Mapping[str, object], name: str) -> int:
    value = data[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise JaxCheckpointError(f"model_config.{name} must be an integer, got {type(value).__name__}")
    return value


def _require_bool(data: Mapping[str, object], name: str) -> bool:
    value = data[name]
    if not isinstance(value, bool):
        raise JaxCheckpointError(f"model_config.{name} must be a boolean, got {type(value).__name__}")
    return value


def _require_str(data: Mapping[str, object], name: str) -> str:
    value = data[name]
    if not isinstance(value, str):
        raise JaxCheckpointError(f"model_config.{name} must be a string, got {type(value).__name__}")
    return value


def _validate_config(config: GPTConfig) -> None:
    positive = {
        "sequence_len": config.sequence_len,
        "vocab_size": config.vocab_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_kv_head": config.n_kv_head,
        "n_embd": config.n_embd,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise JaxCheckpointError(f"model_config dimensions must be positive: {invalid}")
    if config.n_embd % config.n_head != 0:
        raise JaxCheckpointError("model_config.n_embd must be divisible by n_head")
    if config.n_head % config.n_kv_head != 0:
        raise JaxCheckpointError("model_config.n_kv_head must divide n_head")
    if (config.n_embd // config.n_head) % 2 != 0:
        raise JaxCheckpointError("model_config head dimension must be even for rotary embeddings")
    if config.attention_type not in {"standard", "tropical", "ultrametric"}:
        raise JaxCheckpointError(f"unsupported model_config.attention_type: {config.attention_type!r}")
    if config.optimizer_type not in {"adamw", "hoss"}:
        raise JaxCheckpointError(f"unsupported model_config.optimizer_type: {config.optimizer_type!r}")
    if config.use_tropical:
        raise JaxCheckpointError("model_config.use_tropical is not an active JAX architecture field and must be false")
    if config.init_cache:
        raise JaxCheckpointError("serving checkpoints require model_config.init_cache=false")


def _config_from_mapping(data: Mapping[str, object]) -> GPTConfig:
    expected = {field.name for field in fields(GPTConfig)}
    _require_exact_keys(data, expected, "model_config")
    config = GPTConfig(
        sequence_len=_require_int(data, "sequence_len"),
        vocab_size=_require_int(data, "vocab_size"),
        n_layer=_require_int(data, "n_layer"),
        n_head=_require_int(data, "n_head"),
        n_kv_head=_require_int(data, "n_kv_head"),
        n_embd=_require_int(data, "n_embd"),
        use_tropical=_require_bool(data, "use_tropical"),
        attention_type=_require_str(data, "attention_type"),
        optimizer_type=_require_str(data, "optimizer_type"),
        init_cache=_require_bool(data, "init_cache"),
    )
    _validate_config(config)
    return config


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise JaxCheckpointError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _read_manifest(checkpoint_dir: Path) -> tuple[GPTConfig, int, str, int]:
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    try:
        raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JaxCheckpointError(
            f"missing {manifest_path}; an incomplete directory is not a published JAX serving checkpoint"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise JaxCheckpointError(f"could not read JAX checkpoint manifest {manifest_path}: {exc}") from exc

    manifest = _mapping(raw_manifest, "manifest")
    _require_exact_keys(
        manifest,
        {"schema", "architecture", "step", "model_config", "variables", "tokenizer"},
        "manifest",
    )
    if manifest["schema"] != CHECKPOINT_SCHEMA:
        raise JaxCheckpointError(
            f"unsupported checkpoint schema {manifest['schema']!r}; expected {CHECKPOINT_SCHEMA!r}"
        )
    if manifest["architecture"] != CHECKPOINT_ARCHITECTURE:
        raise JaxCheckpointError(
            f"checkpoint architecture {manifest['architecture']!r} does not match {CHECKPOINT_ARCHITECTURE!r}"
        )
    step = manifest["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise JaxCheckpointError("manifest.step must be a non-negative integer")

    variables = _mapping(manifest["variables"], "manifest.variables")
    _require_exact_keys(variables, {"format", "path"}, "manifest.variables")
    if variables != {"format": VARIABLES_FORMAT, "path": VARIABLES_FILENAME}:
        raise JaxCheckpointError("manifest.variables does not match the v1 variables contract")

    tokenizer = _mapping(manifest["tokenizer"], "manifest.tokenizer")
    _require_exact_keys(tokenizer, {"format", "path", "sha256", "vocab_size"}, "manifest.tokenizer")
    expected_tokenizer = f"{TOKENIZER_DIRECTORY}/{TOKENIZER_FILENAME}"
    if tokenizer["format"] != TOKENIZER_FORMAT or tokenizer["path"] != expected_tokenizer:
        raise JaxCheckpointError("manifest.tokenizer does not match the v1 tokenizer contract")
    tokenizer_vocab_size = tokenizer["vocab_size"]
    if not isinstance(tokenizer_vocab_size, int) or isinstance(tokenizer_vocab_size, bool):
        raise JaxCheckpointError("manifest.tokenizer.vocab_size must be an integer")
    tokenizer_sha256 = tokenizer["sha256"]
    if (
        not isinstance(tokenizer_sha256, str)
        or len(tokenizer_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tokenizer_sha256)
    ):
        raise JaxCheckpointError("manifest.tokenizer.sha256 must be a lowercase SHA-256 digest")

    config = _config_from_mapping(_mapping(manifest["model_config"], "manifest.model_config"))
    return config, tokenizer_vocab_size, tokenizer_sha256, step


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise JaxCheckpointError(f"could not hash checkpoint artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _array_signature(value: object) -> tuple[tuple[int, ...], np.dtype[np.generic]] | None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return None
    return tuple(int(dimension) for dimension in shape), np.dtype(dtype)


def _validate_state_tree(template: object, restored: object, path: str = "variables") -> None:
    if isinstance(template, Mapping):
        if not isinstance(restored, Mapping):
            raise JaxCheckpointError(f"{path} must be a mapping")
        template_keys = set(template)
        restored_keys = set(restored)
        if template_keys != restored_keys:
            missing = sorted(str(key) for key in template_keys - restored_keys)
            unexpected = sorted(str(key) for key in restored_keys - template_keys)
            raise JaxCheckpointError(f"{path} keys mismatch: missing={missing}, unexpected={unexpected}")
        for key in template:
            _validate_state_tree(template[key], restored[key], f"{path}.{key}")
        return

    if isinstance(template, Sequence) and not isinstance(template, (str, bytes, bytearray)):
        if not isinstance(restored, Sequence) or isinstance(restored, (str, bytes, bytearray)):
            raise JaxCheckpointError(f"{path} must be a sequence")
        if len(template) != len(restored):
            raise JaxCheckpointError(f"{path} length mismatch: expected {len(template)}, got {len(restored)}")
        for index, (template_item, restored_item) in enumerate(zip(template, restored)):
            _validate_state_tree(template_item, restored_item, f"{path}[{index}]")
        return

    template_signature = _array_signature(template)
    restored_signature = _array_signature(restored)
    if template_signature is not None:
        if restored_signature is None:
            raise JaxCheckpointError(f"{path} must be an array")
        if template_signature != restored_signature:
            raise JaxCheckpointError(
                f"{path} shape/dtype mismatch: expected {template_signature}, got {restored_signature}"
            )
        return
    if type(restored) is not type(template):
        raise JaxCheckpointError(
            f"{path} type mismatch: expected {type(template).__name__}, got {type(restored).__name__}"
        )


def _model_template(config: GPTConfig) -> tuple[GPT, VariableDict]:
    model = GPT(config)
    dummy_input = jnp.zeros((1, 1), dtype=jnp.int32)
    try:
        variables = cast(VariableDict, model.init(jax.random.PRNGKey(0), dummy_input, train=False))
    except (TypeError, ValueError) as exc:
        raise JaxCheckpointError(f"invalid JAX GPT architecture metadata: {exc}") from exc
    return model, variables


def _validate_forward(model: GPT, variables: VariableDict, config: GPTConfig) -> None:
    try:
        raw_logits: object = model.apply(variables, jnp.zeros((1, 1), dtype=jnp.int32), train=False)
    except (TypeError, ValueError) as exc:
        raise JaxCheckpointError(f"checkpoint dry-run inference failed: {exc}") from exc
    if not isinstance(raw_logits, jax.Array) or raw_logits.shape != (1, 1, config.vocab_size):
        shape = getattr(raw_logits, "shape", None)
        raise JaxCheckpointError(
            f"checkpoint dry-run returned logits shape {shape}, expected {(1, 1, config.vocab_size)}"
        )
    if not bool(np.asarray(jnp.all(jnp.isfinite(raw_logits))).item()):
        raise JaxCheckpointError("checkpoint dry-run returned non-finite logits")


def write_serving_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    config: GPTConfig,
    variables: VariableDict,
    tokenizer: HuggingFaceTokenizer,
) -> Path:
    """Publish a new, self-contained JAX serving checkpoint.

    The target directory must not exist. This avoids silently overwriting a
    checkpoint and allows ``manifest.json`` to be the final publication marker.
    """

    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise JaxCheckpointError("checkpoint step must be a non-negative integer")
    _validate_config(config)
    tokenizer_vocab_size = int(tokenizer.get_vocab_size())
    if not (0 < tokenizer_vocab_size <= config.vocab_size):
        raise JaxCheckpointError(
            f"tokenizer vocabulary has {tokenizer_vocab_size} entries but model_config.vocab_size is {config.vocab_size}"
        )
    model, template = _model_template(config)
    template_state = serialization.to_state_dict(template)
    variables_state = serialization.to_state_dict(variables)
    _validate_state_tree(template_state, variables_state)
    _validate_forward(model, variables, config)
    encoded_variables = serialization.to_bytes(variables)

    destination = Path(checkpoint_dir)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise JaxCheckpointError(f"refusing to overwrite existing checkpoint directory {destination}") from exc
    except OSError as exc:
        raise JaxCheckpointError(f"could not create checkpoint directory {destination}: {exc}") from exc

    variables_path = destination / VARIABLES_FILENAME
    with variables_path.open("xb") as variables_file:
        variables_file.write(encoded_variables)
    tokenizer.save(destination / TOKENIZER_DIRECTORY)
    tokenizer_path = destination / TOKENIZER_DIRECTORY / TOKENIZER_FILENAME
    tokenizer_sha256 = _sha256_file(tokenizer_path)

    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "step": step,
        "model_config": asdict(config),
        "variables": {"format": VARIABLES_FORMAT, "path": VARIABLES_FILENAME},
        "tokenizer": {
            "format": TOKENIZER_FORMAT,
            "path": f"{TOKENIZER_DIRECTORY}/{TOKENIZER_FILENAME}",
            "sha256": tokenizer_sha256,
            "vocab_size": tokenizer_vocab_size,
        },
    }
    manifest_path = destination / MANIFEST_FILENAME
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    return destination


def load_serving_checkpoint(checkpoint_dir: str | Path) -> JaxServingCheckpoint:
    """Load and fully validate a JAX serving checkpoint before publication."""

    source = Path(checkpoint_dir)
    if not source.is_dir():
        raise JaxCheckpointError(f"JAX checkpoint directory does not exist: {source}")
    config, declared_tokenizer_vocab_size, declared_tokenizer_sha256, step = _read_manifest(source)

    tokenizer_path = source / TOKENIZER_DIRECTORY
    tokenizer_file = tokenizer_path / TOKENIZER_FILENAME
    actual_tokenizer_sha256 = _sha256_file(tokenizer_file)
    if not hmac.compare_digest(declared_tokenizer_sha256, actual_tokenizer_sha256):
        raise JaxCheckpointError(
            "checkpoint tokenizer digest does not match its manifest: "
            f"manifest={declared_tokenizer_sha256}, actual={actual_tokenizer_sha256}"
        )
    try:
        tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_path)
    except Exception as exc:
        raise JaxCheckpointError(f"could not load checkpoint tokenizer from {tokenizer_path}: {exc}") from exc
    actual_tokenizer_vocab_size = int(tokenizer.get_vocab_size())
    if declared_tokenizer_vocab_size != actual_tokenizer_vocab_size:
        raise JaxCheckpointError(
            "checkpoint tokenizer vocabulary does not match its manifest: "
            f"manifest={declared_tokenizer_vocab_size}, actual={actual_tokenizer_vocab_size}"
        )
    if not (0 < actual_tokenizer_vocab_size <= config.vocab_size):
        raise JaxCheckpointError(
            "checkpoint tokenizer vocabulary is incompatible with model_config: "
            f"tokenizer={actual_tokenizer_vocab_size}, model={config.vocab_size}"
        )

    variables_path = source / VARIABLES_FILENAME
    try:
        encoded_variables = variables_path.read_bytes()
    except OSError as exc:
        raise JaxCheckpointError(f"could not read checkpoint variables {variables_path}: {exc}") from exc
    if not encoded_variables:
        raise JaxCheckpointError(f"checkpoint variables file is empty: {variables_path}")

    model, template = _model_template(config)
    try:
        restored_state: object = serialization.msgpack_restore(encoded_variables)
    except Exception as exc:
        raise JaxCheckpointError(f"could not decode checkpoint variables {variables_path}: {exc}") from exc
    template_state = serialization.to_state_dict(template)
    _validate_state_tree(template_state, restored_state)
    restored_mapping = cast(dict[str, Any], _mapping(restored_state, "variables"))
    try:
        variables = cast(VariableDict, serialization.from_state_dict(template, restored_mapping))
    except (TypeError, ValueError, KeyError) as exc:
        raise JaxCheckpointError(f"could not restore checkpoint variables {variables_path}: {exc}") from exc
    _validate_forward(model, variables, config)
    return JaxServingCheckpoint(
        checkpoint_dir=source,
        step=step,
        model=model,
        variables=variables,
        tokenizer=tokenizer,
        config=config,
    )
