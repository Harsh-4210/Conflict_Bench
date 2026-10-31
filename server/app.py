"""
FastAPI application for the ConflictBench Environment.

Endpoints (openenv-core 0.2.3):
    - POST /reset: Reset (stateless: each HTTP call gets a fresh environment)
    - POST /step: Execute an action (stateless, see above)
    - GET /state, GET /schema, GET /metadata, GET /health
    - WS /ws: persistent session; use this for real reset-then-step episodes

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    python -m server.app --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from .conflictbench_environment import ConflictBenchEnvironment
    from .models import ConflictBenchAction, ConflictBenchObservation
except ImportError:
    from server.conflictbench_environment import ConflictBenchEnvironment
    from server.models import ConflictBenchAction, ConflictBenchObservation


# The environment class is the factory: one fresh, isolated instance per session.
# reveal_ground_truth defaults to False, so served observations never leak answers.
app = create_app(
    ConflictBenchEnvironment,
    ConflictBenchAction,
    ConflictBenchObservation,
    env_name="conflictbench",
    max_concurrent_envs=10,
)


def main():
    """Entry point for direct execution: python -m server.app [--host H] [--port P]."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
