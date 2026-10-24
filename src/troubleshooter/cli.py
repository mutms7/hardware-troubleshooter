import argparse
import asyncio

from .settings import Settings


def main():
    parser = argparse.ArgumentParser(description="Hardware measurement MCP boundary")
    sub = parser.add_subparsers(dest="command", required=True)
    mcp = sub.add_parser("mcp", help="Run the local MCP stdio server")
    mcp.add_argument("--backend", choices=["none", "stub"])
    mcp.add_argument("--scenario", help="Synthetic scenario, requires --backend stub")
    smoke = sub.add_parser("smoke", help="Run one complete simulated test through real MCP")
    smoke.add_argument(
        "--scenario",
        choices=["success", "contact_failed", "board_moved", "stage_timeout"],
        default="success",
    )
    args = parser.parse_args()
    if args.command == "smoke":
        from .smoke import run_smoke

        asyncio.run(run_smoke(args.scenario))
        return
    overrides = {}
    if args.backend:
        overrides["backend"] = args.backend
    if args.scenario:
        if args.backend != "stub":
            parser.error("--scenario requires explicit --backend stub")
        overrides["stub_scenario"] = args.scenario
    from .mcp_server import main as run

    run(Settings(**overrides))


if __name__ == "__main__":
    main()
