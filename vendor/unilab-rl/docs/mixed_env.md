# Mixed Backend Environment Runtime

`uni_rl.ipc.mixed_env.MixedProcessEnv` joins injected vector environments into
one fixed-size batch. The policy stays in the caller process. Each active
`WorkerSpec` starts a separate `spawn` process and exchanges numeric payloads
through shared host memory; backend factories must be importable top-level
callables and defer engine/CUDA imports until invocation.

Create a `WorkerSpec(name, factory, ...)` for every declared backend, including
initially inactive backends. Pass an exact name-to-integer-count mapping to
`MixedProcessEnv(specs, counts)`. `init_state()`, `step(actions)` and
`reset(indices)` follow `uni_rl.env_contract`; reset returns selected rows in
the requested index order. Always use a context manager or call `close()`.

Use `process_env` for worker-specific environment variables, `cpu_ids` for
CPU affinity, `timeout_s` for startup and per-operation deadlines, and an
injected `metadata_factory` for application-owned semantic checks. Startup and
operation deadlines are enforced independently for each pending backend; waiting
for a slow backend cannot hide another backend's expired deadline. Control
periods, episode durations, ordered observation groups, action bounds and semantic metadata must
match exactly across workers. Action-space bounds may be infinite (an unbounded
Gym Box); NaN bounds, invalid shapes and inverted bounds are rejected.
Floating-point sample transport is explicitly float32;
invalid shapes, nonfinite payloads and missing timeout final observations fail
closed. Factories remain responsible for seeding their own engine runtime.

`reconfigure(counts)` keeps the total batch size unchanged, rebuilds all workers
only when integer counts change, increments `generation`, and clears physical
episode state. Call `init_state()` again after a rebuild. `get_stats()` returns
cumulative per-backend counters, including samples and interrupted episodes.
This class does not implement policy updates, ratio allocation, stage scheduling
or checkpointing; those belong to the algorithm and application layers.

From the unilab-rl checkout, run the focused tests and checks:

```bash
uv run pytest tests/test_mixed_env.py
uv run ruff check src/uni_rl/ipc/mixed_env.py tests/test_mixed_env.py
uv run ruff format --check src/uni_rl/ipc/mixed_env.py tests/test_mixed_env.py
```

For an already provisioned development interpreter without syncing this
checkout's dependency lock, use:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  uv run --cache-dir /tmp/unilab-test-cache --no-project \
  --python ../.venv-test/bin/python \
  python -m pytest tests/test_mixed_env.py
```

The optional plugin-autoload switch isolates these unit tests from globally
installed ROS pytest plugins with incompatible pytest hook versions.
