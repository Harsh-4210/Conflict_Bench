"""
FastAPI application for the ConflictBench Environment.

This module creates an HTTP server that exposes the ConflictBenchEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from .conflictbench_environment import ConflictBenchEnvironment
except ImportError:
    from server.conflictbench_environment import ConflictBenchEnvironment

from openenv.core.env_server.types import TextAction, TextObservation


# Create the app with web interface and README integration
app = create_app(
    ConflictBenchEnvironment,
    TextAction,
    TextObservation,
    env_name="conflictbench",
    max_concurrent_envs=10,
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m conflictbench.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn conflictbench.server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
