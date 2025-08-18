# Model-Guided Research Test Suite

This directory contains comprehensive testing tools for the model-guided research codebase, featuring beautiful rich terminal output and extensive validation capabilities.

## Test Runners

### 1. **test_runner.py** - Standard Test Suite
The main test runner with code quality checks harmonized with recent linting fixes.

```bash
# Basic usage
python tests/test_runner.py

# Skip code quality checks
python tests/test_runner.py --no-quality

# Skip JAX diagnostics
python tests/test_runner.py --no-jax

# Full testing
python tests/test_runner.py
```

**Features:**
- ✅ Module import validation
- ✅ Code quality checks (verifies linting fixes)
- ✅ Demo function existence verification
- ✅ Documentation validation
- ✅ JAX environment diagnostics
- ✅ Beautiful rich terminal output with progress bars
- ✅ Detailed failure reporting

### 2. **test_runner_enhanced.py** - Advanced Test Suite
Enhanced test runner with live updates, memory monitoring, and smoke testing capabilities.

```bash
# Basic usage
python tests/test_runner_enhanced.py

# Run with actual demo execution (smoke tests)
python tests/test_runner_enhanced.py --demos

# Verbose output with detailed reports
python tests/test_runner_enhanced.py --verbose --demos

# Set custom timeout for demo execution
python tests/test_runner_enhanced.py --demos --timeout 60

# Save HTML report
python tests/test_runner_enhanced.py --save-report report.html
```

**Features:**
- 🔥 Smoke testing with actual demo execution
- 📊 Live dashboard with real-time updates
- 💾 Memory usage monitoring
- ⏱️ Performance metrics and timing
- 📈 Progress bars with ETA
- 🎨 Beautiful layouts and tree displays
- 📝 HTML report generation
- ⚡ Parallel test capability

**Requirements:**
```bash
pip install psutil  # Required for enhanced runner
```

## Test Coverage

### Phase 1: Import Tests
- Validates all modules can be imported
- Measures import time
- Catches missing dependencies

### Phase 2: Code Quality (Standard Runner)
Verifies that linting fixes have been applied:
- ✅ No ambiguous variable names (l, O)
- ✅ No old-style percent formatting
- ✅ Proper exception chaining
- ✅ No unused loop variables

### Phase 3: Function Validation
- Checks for required `demo()` functions
- Validates function signatures
- Ensures callability

### Phase 4: Documentation
- Verifies markdown documentation exists
- Checks documentation quality (enhanced runner)
- Validates file sizes

### Phase 5: Smoke Tests (Enhanced Runner Only)
- Executes demo functions with timeout
- Monitors memory usage
- Captures output and errors
- Validates execution success

## Code Quality Checks

The test suite validates the following code improvements:

1. **Variable Naming**
   - `l` → `loss_val`, `loss_func`, `init_weight`
   - `O` → `num_offsets`
   - `j` → `_j` (for unused variables)

2. **String Formatting**
   - `% ()` → f-strings

3. **Exception Handling**
   - `raise typer.Exit(1)` → `raise typer.Exit(1) from e`

4. **Type Annotations**
   - Proper type hints for JAX arrays
   - Fixed mypy issues

## Installation

### Basic Requirements
```bash
# Core dependencies (already in pyproject.toml)
pip install rich typer
```

### Enhanced Runner Requirements
```bash
# Install test requirements
pip install -r tests/requirements-test.txt

# Or individually
pip install psutil rich
```

## Output Examples

### Standard Runner Output
```
🧮 Model-Guided Research Test Suite 🧮
Version 1.1 - Harmonized with latest code changes

Phase 1: Testing Module Imports
✅ IFS/Fractal
✅ Knot/Braid
✅ Matrix/Gauge
...

Phase 2: Code Quality Checks
✅ All modules clean
...

Test Summary
┌─────────────┬────────┬────────┬───────┐
│ Category    │ Passed │ Failed │ Total │
├─────────────┼────────┼────────┼───────┤
│ Imports     │ 11     │ 0      │ 11    │
│ Code Quality│ 11     │ 0      │ 11    │
│ Functions   │ 11     │ 0      │ 11    │
│ Docs        │ 11     │ 0      │ 11    │
└─────────────┴────────┴────────┴───────┘

✅ ALL TESTS PASSED!
44/44 tests successful in 2.34s
```

### Enhanced Runner Live Display
```
╔══════════════════════════════════════════╗
║ 🧮 Model-Guided Research Enhanced Suite  ║
╚══════════════════════════════════════════╝

Overall Progress
Running tests... ━━━━━━━━━━━━━━━ 75% 0:00:15

Current Tests
Smoke testing... ━━━━━━━━━━━━━━━ 45% 0:00:08

🔬 Test Execution Details
├── 📦 Phase 1: Import Tests
│   ├── ✅ IFS/Fractal: 0.123s
│   ├── ✅ Knot/Braid: 0.089s
│   └── ...
├── 🔥 Phase 5: Smoke Tests
│   ├── ✅ IFS/Fractal: 1.23s (45.2MB)
│   └── ⚠️ Knot/Braid: 2.34s (high memory)

CPU: 23.4% | Memory: 45.2% | Time: 14:32:15
```

## Continuous Integration

For CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run Tests
  run: |
    python tests/test_runner.py --no-jax
    
- name: Run Enhanced Tests
  run: |
    pip install psutil
    python tests/test_runner_enhanced.py --demos --timeout 30
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `uv pip install -e .`
   - Check Python version compatibility (3.10+)

2. **JAX Issues**
   - JAX not installed: `pip install jax jaxlib`
   - GPU/TPU setup: Check JAX documentation

3. **Memory Issues (Enhanced Runner)**
   - Reduce timeout: `--timeout 10`
   - Skip smoke tests: remove `--demos` flag

4. **Code Quality Failures**
   - Run linting: `ruff check --fix`
   - Check specific issues in the output

## Development

To add new tests:

1. Add module name to `DEMO_MODULES` list
2. Ensure module has a `demo()` function
3. Create documentation in `markdown_documentation/`
4. Run tests to verify

## Version History

- **v1.0**: Original test runner with basic checks
- **v1.1**: Harmonized with linting fixes, added code quality checks
- **v2.0**: Enhanced runner with live updates and smoke testing

## License

Part of the Model-Guided Research project.