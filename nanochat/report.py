"""
Utilities for generating training report cards. More messy code than usual, will fix.
"""

import datetime
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess  # nosec B404
from typing import Any

import psutil

from nanochat.torch_imports import torch

ALLOWED_CMDS = {"git", "files-to-prompt"}


def run_command(cmd):
    """Run a whitelisted shell command and return output, or None if it fails."""
    try:
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        if not cmd or cmd[0] not in ALLOWED_CMDS:
            return None
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, timeout=5)  # nosec B603 safe allowlist
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def get_git_info():
    """Get current git commit, branch, and dirty status."""
    info = {}
    info["commit"] = run_command("git rev-parse --short HEAD") or "unknown"
    info["commit_full"] = run_command("git rev-parse HEAD") or "unknown"
    info["branch"] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # Check if repo is dirty (has uncommitted changes)
    status = run_command("git status --porcelain")
    info["dirty"] = bool(status) if status is not None else False

    # Get commit message
    info["message"] = run_command("git log -1 --pretty=%B") or ""
    info["message"] = info["message"].split("\n")[0][:80]  # First line, truncated

    return info


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    names: list[str] = []
    memory_gb: list[float] = []
    info = {
        "available": True,
        "count": num_devices,
        "names": names,
        "memory_gb": memory_gb,
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        names.append(props.name)
        memory_gb.append(props.total_memory / (1024**3))

    # Get CUDA version
    info["cuda_version"] = torch.version.cuda or "unknown"

    return info


def get_system_info():
    """Get system information."""
    info = {}

    # Basic system info
    info["hostname"] = socket.gethostname()
    info["platform"] = platform.system()
    info["python_version"] = platform.python_version()
    info["torch_version"] = torch.__version__

    # CPU and memory
    info["cpu_count"] = psutil.cpu_count(logical=False)
    info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    info["memory_gb"] = psutil.virtual_memory().total / (1024**3)

    # User and environment
    info["user"] = os.environ.get("USER", "unknown")
    from nanochat.common import get_base_dir

    info["nanochat_base_dir"] = get_base_dir()
    info["nanochat_base_dir_env"] = os.environ.get("NANOCHAT_BASE_DIR")
    info["working_dir"] = os.getcwd()

    return info


# -----------------------------------------------------------------------------
# Per-step metrics stream (bead rz8.2): one JSONL record per logged step, the
# durable data source for scaling fits (E2), verdict adjudication (G2), the
# dashboard (nyp), and regression forensics. Schema documented in
# artifacts/README.md.

METRICS_SCHEMA_VERSION = "mgr.metrics.v1"


def build_provenance(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Tamper-evidence block for artifacts used as adjudication evidence.

    git_dirty=true (or no git at all) marks the artifact TAINTED: fine for
    exploration, but the G2 verdict engine refuses tainted evidence — results
    from a working tree no reviewer can reconstruct are unattributable to any
    code state.
    """
    import hashlib
    import json as json_mod

    git = get_git_info()
    sha = git.get("commit_full") or "unknown"
    git_available = sha != "unknown"
    dirty = bool(git.get("dirty", False))
    config_hash = hashlib.sha256(
        json_mod.dumps(config_dict, sort_keys=True, default=str).encode()
    ).hexdigest()
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "git_sha": sha if git_available else None,
        "git_dirty": dirty if git_available else None,
        "config_hash": config_hash,
        "data_snapshot_hash": None,  # wiz bead's helper, when it lands
        "tainted": (not git_available) or dirty,
    }


class MetricsStream:
    """Buffered JSONL writer for per-step training metrics.

    The header line (record type "header", carrying the provenance block) is
    written and flushed IMMEDIATELY so even a crashed run leaves an
    attributable artifact; step records buffer in memory and flush every
    `flush_every` records, at val evaluations, and on close() — the caller
    closes inside try/finally so KeyboardInterrupt still lands the buffer.

    A RESUMED run must pass append=True: the stream then appends to the
    existing file instead of truncating it (a resume previously erased the
    parent process's entire step history — found by the rz8.8 e2e resume
    scenario), and marks the splice with a "resume_header" record carrying
    the resuming process's provenance, so each segment of the trajectory
    stays attributable to its commit.
    """

    def __init__(self, path: Any, *, provenance: dict[str, Any], flush_every: int = 50, append: bool = False):
        import json as json_mod

        self._json = json_mod
        self._flush_every = max(1, int(flush_every))
        self._buffer: list[str] = []
        mode = "a" if append else "w"
        self._fh = open(path, mode, encoding="utf-8")  # noqa: SIM115 - lifetime managed by close()
        header = {"type": "resume_header" if append else "header", **provenance}
        self._fh.write(self._json.dumps(header, sort_keys=True) + "\n")
        self._fh.flush()

    def write(self, record: dict[str, Any]) -> None:
        self._buffer.append(self._json.dumps(record, sort_keys=True))
        if len(self._buffer) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            self._fh.write("\n".join(self._buffer) + "\n")
            self._buffer.clear()
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self.flush()
            self._fh.close()


def read_metrics_jsonl(path: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    """Read a metrics.jsonl stream. Returns (header, records, problems).

    Malformed lines are skipped and reported in `problems` — one bad line must
    never crash an analysis (E2/G2 consume these files programmatically).
    """
    import json as json_mod

    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return None, [], [f"cannot read {path}: {exc}"]
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json_mod.loads(line)
        except json_mod.JSONDecodeError as exc:
            problems.append(f"line {lineno}: malformed JSON ({exc})")
            continue
        if not isinstance(obj, dict) or "type" not in obj:
            problems.append(f"line {lineno}: record must be a mapping with a 'type' field")
            continue
        if obj["type"] == "header":
            if header is not None:
                problems.append(f"line {lineno}: duplicate header")
            elif obj.get("schema_version") != METRICS_SCHEMA_VERSION:
                problems.append(
                    f"line {lineno}: schema_version {obj.get('schema_version')!r} != {METRICS_SCHEMA_VERSION!r}"
                )
                header = obj
            else:
                header = obj
        else:
            records.append(obj)
    if header is None:
        problems.append("no header line found")
    return header, records, problems


# ---------------------------------------------------------------------------
# Live training dashboard (bead nyp). Renders the SAME record dicts that
# MetricsStream writes - no second telemetry path, no overhead when off (the
# trainer simply never constructs one). Mechanism diagnostics (route_coverage,
# braid_q1/q2, symplectic_*, certs.*) are auto-adopted: any numeric scalar in
# a step record that is not a core key becomes a tracked sparkline with
# trailing-band drift highlighting - the invariant-drift panel costs nothing
# here because the telemetry beads produce the data; this only renders it.
# ---------------------------------------------------------------------------

_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_DASH_CORE_KEYS = frozenset(
    {
        "type",
        "step",
        "loss",
        "lr",
        "lr_groups",
        "grad_norm",
        "tokens_per_s",
        "tflops",
        "peak_mem_gb",
        "elapsed_s",
        "val_loss",
        "train_loss",
    }
)


def sparkline(values: list[float], width: int = 32) -> str:
    """Unicode block sparkline of the trailing `width` values; non-finite
    points render as '✕' so a NaN excursion stays visible in the strip."""
    import math as math_mod

    tail = list(values)[-width:]
    if not tail:
        return ""
    finite = [v for v in tail if math_mod.isfinite(v)]
    if not finite:
        return "✕" * len(tail)
    lo, hi = min(finite), max(finite)
    span = hi - lo
    out = []
    for v in tail:
        if not math_mod.isfinite(v):
            out.append("✕")
        elif span <= 0:
            out.append(_SPARK_CHARS[3])
        else:
            idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
            out.append(_SPARK_CHARS[max(0, min(idx, len(_SPARK_CHARS) - 1))])
    return "".join(out)


def _drift_flag(series: list[float], min_history: int = 8, k: float = 3.0) -> bool:
    """True when the latest value departs the trailing band (mean ± k·std of
    the PRIOR points) - the warn highlight of the invariant-drift panel."""
    import math as math_mod
    import statistics as stats_mod

    finite = [v for v in series if math_mod.isfinite(v)]
    if len(finite) < min_history:
        return False
    prior, last = finite[:-1], finite[-1]
    mu = stats_mod.mean(prior)
    sd = stats_mod.stdev(prior)
    if sd <= 0 or not math_mod.isfinite(sd):
        return False
    return abs(last - mu) > k * sd


# bead 92m: per-head heatmap rendering for the dashboard. A per-head metric is a
# short numeric vector (one entry per attention head, length 2..64) carried in a
# step record under a "head"-named key (tropical_gamma_head_mean, etc.).
_HEAT_RAMP = "·░▒▓█"  # lowest is a dot (not a blank) so a row keeps its shape


def _is_head_vector(val: Any) -> bool:
    if not isinstance(val, (list, tuple)) or not (2 <= len(val) <= 64):
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val)


def _head_heat_row(values: list[float]) -> Any:
    """A single colored heat strip for a per-head vector (self-scaled).

    Each head becomes a shaded, 256-color cell (red=low → green=high within the
    row), so route specialization across heads reads at a glance in the terminal
    and in the recorded HTML export.
    """
    import math as math_mod

    from rich.text import Text

    finite = [v for v in values if math_mod.isfinite(v)]
    lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
    span = (hi - lo) or 1.0
    out = Text()
    for v in values:
        if not math_mod.isfinite(v):
            out.append(" ✕ ", style="bold red")
            continue
        frac = (v - lo) / span
        ch = _HEAT_RAMP[min(len(_HEAT_RAMP) - 1, int(frac * (len(_HEAT_RAMP) - 1) + 0.5))]
        # 256-color cube ramp red(196)=(5,0,0) -> green(46)=(0,5,0): each step
        # drops a red unit (-36) and adds a green unit (+6), net -30 per step.
        color = 196 - 30 * int(round(frac * 5))
        out.append(f"{ch}", style=f"color({max(16, min(231, color))})")
    return out


def _head_diversity(values: list[float]) -> float | None:
    """Cross-head spread (population stdev) - a cheap route-diversity scalar."""
    import math as math_mod
    import statistics as stats_mod

    finite = [v for v in values if math_mod.isfinite(v)]
    if len(finite) < 2:
        return None
    return float(stats_mod.pstdev(finite))


class TrainingDashboard:
    """Rich live console dashboard for a training run (bead nyp).

    Usage from the trainer (rank 0 only):
        dash = TrainingDashboard(run_id=..., mechanism=..., max_steps=..., flags={...},
                                 html_path=artifacts/dashboard/<run_id>.html)
        dash.start()
        ... dash.observe(record) right after metrics_stream.write(record) ...
        dash.close(results)   # stops the live view; writes the HTML export

    Rendering is pull-free: observe() pushes the record and refreshes the
    live view, so a disabled dashboard (never constructed) costs nothing.
    """

    def __init__(
        self,
        *,
        run_id: str,
        mechanism: str,
        max_steps: int,
        flags: dict[str, Any] | None = None,
        html_path: Any = None,
        console: Any = None,
        window: int = 256,
        show_head_heatmaps: bool = True,
    ):
        from collections import deque

        from rich.console import Console

        self.run_id = run_id
        self.mechanism = mechanism
        self.max_steps = max(1, int(max_steps))
        self.flags = dict(flags or {})
        self.html_path = html_path
        self.console = console or Console()
        self._window = int(window)
        self._loss: deque[float] = deque(maxlen=self._window)
        self._grad: deque[float] = deque(maxlen=self._window)
        self._toks: deque[float] = deque(maxlen=self._window)
        self._tflops: deque[float] = deque(maxlen=self._window)
        self._val: list[tuple[int, float]] = []
        self._diags: dict[str, deque[float]] = {}
        # bead 92m: per-head metric vectors (e.g. tropical_gamma_head_mean,
        # attn_entropy_head_mean) -> latest snapshot, rendered as a route
        # diversity/entropy heatmap. Data-gated: the panel only appears when a
        # per-head field actually flows, so standard runs pay nothing.
        self.show_head_heatmaps = bool(show_head_heatmaps)
        self._head_latest: dict[str, list[float]] = {}
        self._head_step: dict[str, int] = {}
        self._step = 0
        self._latest: dict[str, Any] = {}
        self._nan_seen = False
        self._live: Any = None

    def start(self) -> None:
        from rich.live import Live

        self._live = Live(self._render(), console=self.console, refresh_per_second=4, transient=False)
        self._live.start()

    def observe(self, record: dict[str, Any]) -> None:
        import math as math_mod
        from collections import deque as _deque

        rtype = record.get("type")
        if rtype == "step":
            self._step = int(record.get("step", self._step))
            self._latest = record
            for key, series in (("loss", self._loss), ("grad_norm", self._grad),
                                ("tokens_per_s", self._toks), ("tflops", self._tflops)):
                val = record.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    series.append(float(val))
                    if not math_mod.isfinite(float(val)):
                        self._nan_seen = True
            for key, val in record.items():
                if key in _DASH_CORE_KEYS:
                    continue
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    self._diags.setdefault(key, _deque(maxlen=self._window)).append(float(val))
                elif self.show_head_heatmaps and "head" in key and _is_head_vector(val):
                    # bead 92m: a per-head metric vector (route margins / entropy)
                    self._head_latest[key] = [float(x) for x in val]
                    self._head_step[key] = self._step
        elif rtype == "val":
            val = record.get("val_loss")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                self._val.append((int(record.get("step", self._step)), float(val)))
        if self._live is not None:
            self._live.update(self._render())

    def close(self, results: dict[str, Any] | None = None) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self.html_path is not None:
            self._export_html(results)

    # -- rendering ----------------------------------------------------------

    def _render(self) -> Any:
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        pct = 100.0 * (self._step + 1) / self.max_steps
        title = Text.assemble(
            (f" {self.run_id} ", "bold white on blue"),
            (f" {self.mechanism} ", "bold black on cyan"),
            (f" step {self._step + 1}/{self.max_steps} ({pct:.0f}%) ", "bold"),
            ((" NaN! ", "bold white on red") if self._nan_seen else (" ✓ finite ", "green")),
        )

        core = Table(box=None, show_header=False, padding=(0, 1))
        core.add_column(justify="right", style="dim")
        core.add_column(justify="left")

        def _fmt(v: Any, spec: str = ".4f") -> str:
            return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else "—"

        core.add_row("loss", f"[bold]{_fmt(self._latest.get('loss'))}[/bold]  [cyan]{sparkline(list(self._loss))}[/cyan]")
        if self._val:
            vstep, vloss = self._val[-1]
            core.add_row("val_ce", f"{vloss:.4f} @ {vstep}  [magenta]{sparkline([v for _, v in self._val])}[/magenta]")
        core.add_row("grad_norm", f"{_fmt(self._latest.get('grad_norm'))}  [yellow]{sparkline(list(self._grad))}[/yellow]")
        core.add_row("lr", _fmt(self._latest.get("lr"), ".2e"))
        core.add_row("tok/s", f"{_fmt(self._latest.get('tokens_per_s'), ',.0f')}  [green]{sparkline(list(self._toks))}[/green]")
        core.add_row("TFLOP/s", f"{_fmt(self._latest.get('tflops'), '.4f')}")
        mem = self._latest.get("peak_mem_gb")
        core.add_row("peak mem", f"{mem:.2f} GB" if isinstance(mem, (int, float)) else "— (cpu)")

        panels: list[Any] = [Panel(core, title=title, border_style="blue")]

        if self.flags:
            flag_text = "  ".join(f"[dim]{k}[/dim]=[bold]{v}[/bold]" for k, v in self.flags.items() if v is not None)
            panels.append(Panel(flag_text, title="feature flags", border_style="dim", height=3))

        if self._diags:
            diag = Table(box=None, show_header=False, padding=(0, 1))
            diag.add_column(justify="right", style="dim", overflow="fold")
            diag.add_column(justify="left")
            for name in sorted(self._diags):
                series = list(self._diags[name])
                drift = _drift_flag(series)
                style = "bold white on red" if drift else "cyan"
                marker = " [bold red]⚠ band[/bold red]" if drift else ""
                diag.add_row(name, f"{series[-1]:.5g}  [{style}]{sparkline(series)}[/{style}]{marker}")
            panels.append(
                Panel(diag, title="mathematical invariants · trailing-band drift watch", border_style="cyan")
            )

        if self.show_head_heatmaps and self._head_latest:
            heat = Table(box=None, show_header=True, header_style="dim", padding=(0, 1))
            heat.add_column("per-head metric", justify="right", style="dim", overflow="fold")
            heat.add_column("heads (low→high)", justify="left")
            heat.add_column("div σ", justify="right", style="dim")
            heat.add_column("μ", justify="right", style="dim")
            for name in sorted(self._head_latest):
                vals = self._head_latest[name]
                div = _head_diversity(vals)
                finite = [v for v in vals if v == v]  # drop NaN
                mu = (sum(finite) / len(finite)) if finite else float("nan")
                label = name.replace("_head_mean", "").replace("_head", "")
                heat.add_row(
                    f"{label} [dim](h0…h{len(vals) - 1})[/dim]",
                    _head_heat_row(vals),
                    f"{div:.4g}" if div is not None else "—",
                    f"{mu:.4g}" if finite else "—",
                )
            panels.append(
                Panel(heat, title="per-head route diversity / entropy · heatmap (bead 92m)", border_style="green")
            )

        return Group(*panels)

    def _export_html(self, results: dict[str, Any] | None) -> None:
        import os

        from rich.console import Console

        rec_console = Console(record=True, width=120, force_terminal=True)
        rec_console.print(self._render())
        if results:
            summary_keys = ("train_ce_final", "val_ce_final", "measured_steps", "measured_tokens")
            line = "  ".join(f"{k}={results[k]}" for k in summary_keys if k in results)
            if line:
                rec_console.print(f"[dim]{line}[/dim]")
        parent = os.path.dirname(str(self.html_path))
        if parent:  # a bare filename has no parent to create
            os.makedirs(parent, exist_ok=True)
        rec_console.save_html(str(self.html_path))


def estimate_cost(gpu_info, runtime_hours=None):
    """Estimate training cost based on GPU type and runtime."""

    # Rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # Try to identify GPU type from name
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info["count"]
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info["count"]  # Default estimate

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None,
    }


def generate_header():
    """Generate the header for a training report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info["branch"]}
- Commit: {git_info["commit"]} {"(dirty)" if git_info["dirty"] else "(clean)"}
- Message: {git_info["message"]}

### Hardware
- Platform: {sys_info["platform"]}
- CPUs: {sys_info["cpu_count"]} cores ({sys_info["cpu_count_logical"]} logical)
- Memory: {sys_info["memory_gb"]:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info["count"]}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info["cuda_version"]}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate: ${cost_info["hourly_rate"]:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info["python_version"]}
- PyTorch: {sys_info["torch_version"]}

"""

    # bloat metrics: package all of the source code and assess its weight
    packaged = run_command('files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml') or ""
    num_chars = len(packaged)
    num_lines = len(packaged.split("\n")) if packaged else 0
    num_files = len([x for x in packaged.split("\n") if x.startswith("<source>")]) if packaged else 0
    num_tokens = num_chars // 4  # assume approximately 4 chars per token

    # count dependencies via uv.lock
    uv_lock_lines = 0
    if os.path.exists("uv.lock"):
        with open("uv.lock", encoding="utf-8") as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header


# -----------------------------------------------------------------------------


def slugify(text):
    """Slugify a text string."""
    return text.lower().replace(" ", "-")


# the expected files and their order
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "midtraining.md",
    "chat-evaluation-mid.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# the metrics we're currently interested in
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]


def extract(section, keys):
    """simple def to extract a single key from a section"""
    if not isinstance(keys, list):
        keys = [keys]  # convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out


def extract_timestamp(content, prefix):
    """Extract timestamp from content with given prefix."""
    for line in content.split("\n"):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
    return None


class Report:
    """Maintains a bunch of logs, generates a final markdown report."""

    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """Log a section of data to the report."""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # directly write the string
                    f.write(item)
                else:
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """Generate the final report."""
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"Generating report to {report_file}")
        final_metrics = {}  # the most important final metrics we'll add as table at the end
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            # write the header first
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # capture bloat data for summary later (the stuff after Bloat header and until \n\n)
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None  # will cause us to not write the total wall clock time
                bloat_data = "[bloat data missing]"
                print(f"Warning: {header_file} does not exist. Did you forget to run `nanochat reset`?")
            # process all the individual sections
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    continue
                with open(section_file, encoding="utf-8") as in_file:
                    section = in_file.read()
                # Extract timestamp from this section (the last section's timestamp will "stick" as end_time)
                if "rl" not in file_name:
                    # Skip RL sections for end_time calculation because RL is experimental
                    end_time = extract_timestamp(section, "timestamp:")
                # extract the most important metrics from the sections
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-mid.md":
                    final_metrics["mid"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K")  # RL only evals GSM8K
                # append this section of the report
                out_file.write(section)
                out_file.write("\n")
            # add the final metrics table
            out_file.write("## Summary\n\n")
            # Copy over the bloat metrics from the header
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # Collect all unique metric names
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # Custom ordering: CORE first, ChatCORE last, rest in middle
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            # Fixed column widths
            stages = ["base", "mid", "sft", "rl"]
            metric_width = 15
            value_width = 8
            # Write table header
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # Write separator
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # Write table rows
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        # also cp the report.md file to current directory
        print("Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """Reset the report."""
        # Remove section files
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove report.md if it exists
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # Generate and write the header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


class DummyReport:
    def log(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass


def get_report():
    # just for convenience, only rank 0 logs to report
    from nanochat.common import get_base_dir, get_dist_info

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate or reset nanochat training reports.")
    parser.add_argument(
        "command",
        nargs="?",
        default="generate",
        choices=["generate", "reset"],
        help="Operation to perform (default: generate)",
    )
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
