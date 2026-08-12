"""Command-line entry point for the segmentation correction tool."""

from __future__ import annotations

import argparse
import asyncio
import logging

from correction_server import (
    DEFAULT_LINK_EXPIRY_SECONDS,
    DEFAULT_PLUGIN_URL,
    start_server,
)

DEFAULT_SERVER_URL = "https://hypha.bioimage.io"


def positive_hours(value: str) -> float:
    """Parse a positive number of hours for the share-link lifetime."""
    hours = float(value)
    if hours <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return hours


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve TIFF images and instance masks through the web correction UI."
    )
    parser.add_argument(
        "directory", help="Directory containing .tif / _masks.tif pairs"
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--mask-suffix", default="_masks.tif")
    parser.add_argument("--corrected-suffix", default="_masks_corrected.tif")
    parser.add_argument("--plugin-url", default=DEFAULT_PLUGIN_URL)
    parser.add_argument(
        "--link-expiry-hours",
        type=positive_hours,
        default=DEFAULT_LINK_EXPIRY_SECONDS / 3600,
        help="share-link lifetime in hours (default: %(default)g)",
    )
    return parser.parse_args(argv)


async def serve(args: argparse.Namespace) -> None:
    server, annotator_url = await start_server(
        args.server_url,
        args.directory,
        mask_suffix=args.mask_suffix,
        corrected_suffix=args.corrected_suffix,
        plugin_url=args.plugin_url,
        link_expiry_seconds=round(args.link_expiry_hours * 3600),
    )
    print("\nCorrection tool is ready. Keep this process running.")
    print(f"Link token expires in {args.link_expiry_hours:g} hours.")
    print("Send this URL to the collaborator:\n")
    print(annotator_url)
    print("\nPress Ctrl-C to stop serving.\n")
    try:
        await asyncio.Event().wait()
    finally:
        await server.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        asyncio.run(serve(parse_args()))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
