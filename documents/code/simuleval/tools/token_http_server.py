#!/usr/bin/env python3
"""Serve a directory over HTTP with a required query token."""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from urllib.parse import parse_qs, urlparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    os.chdir(args.root)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def _authorized(self) -> bool:
            query = parse_qs(urlparse(self.path).query)
            return query.get("token", [""])[0] == args.token

        def do_GET(self) -> None:
            if not self._authorized():
                self.send_error(403)
                return
            self.path = urlparse(self.path).path
            super().do_GET()

        def do_HEAD(self) -> None:
            if not self._authorized():
                self.send_error(403)
                return
            self.path = urlparse(self.path).path
            super().do_HEAD()

    with socketserver.ThreadingTCPServer(("0.0.0.0", args.port), Handler) as httpd:
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
