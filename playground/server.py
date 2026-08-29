"""A local, stdlib-only server for the playground UI.

The repo has never taken a third-party dependency and `submission/README.md`
says so in as many words, so a demo tool is a poor place to take the first
one. This is `http.server` and a hand-written page: no framework, no build
step, no manifest.

One agent instance is built at start-up, which takes a few seconds because it
indexes 50,000 products once, exactly as the scoring path does. It is shared
across requests behind a lock: the agent carries one session's state at a
time by design, and two overlapping requests would interleave two
conversations into it.
"""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import threading
import urllib.parse
import uuid
from pathlib import Path

from evaluator import local_evaluator
from playground import display
from playground import driver
from playground import explain

STATIC = Path(__file__).parent / "static"

DEFAULT_PORT = 8765

# How many products a goal-picker query returns. Enough to choose from, few
# enough that the response stays small.
SEARCH_LIMIT = 25


class Playground:
    """Everything the request handlers share, built once."""

    def __init__(self, catalog_path: str, dataset_path: str) -> None:
        self.lock = threading.Lock()
        self.agent = explain.ExplainingAgent(catalog_path)
        self.products = display.load(catalog_path)
        self.samples = local_evaluator.load_jsonl(dataset_path)
        self.by_id = {row["sample_id"]: row for row in self.samples}
        self.dataset = local_evaluator.catalog_index(catalog_path)
        self.live: driver.Live | None = None

    def sample_list(self) -> list[dict]:
        """The public set as a pickable list, with each target named."""
        rows = []
        for row in self.samples:
            target = str(row["ground_truth"]["parent_asin"])
            rows.append({
                "sample_id": row["sample_id"],
                "scenario_type": row["scenario_type"],
                "target": target,
                "title": display.card(self.products, target)["title"],
            })
        return rows

    def replay(self, sample_id: str) -> dict:
        """Scores one public session and explains it."""
        sample = self.by_id.get(sample_id)
        if sample is None:
            raise KeyError(sample_id)
        with self.lock:
            out = driver.replay(self.agent, sample, self.dataset)
        out["cards"] = self._cards(_asins(out["session"]["turns"]))
        return out

    def open_live(self, profile: dict | None, goal: str | None) -> dict:
        """Starts a free-typed session."""
        with self.lock:
            self.live = driver.Live(
                self.agent, f"playground_{uuid.uuid4().hex}"
            )
            opened = self.live.open(profile, goal)
        opened["goal_card"] = (
            display.card(self.products, goal) if goal else None
        )
        return opened

    def send_live(self, message: str) -> dict:
        """Serves one typed turn."""
        with self.lock:
            if self.live is None:
                raise KeyError("no open session")
            entry = self.live.send(message)
        entry["cards"] = self._cards(_asins([entry]))
        return entry

    def search(self, query: str) -> list[dict]:
        """Finds products by title, for the goal picker."""
        words = query.casefold().split()
        if not words:
            return []
        found = []
        for card in self.products.values():
            haystack = card["title"].casefold()
            if all(word in haystack for word in words):
                found.append(card)
                if len(found) >= SEARCH_LIMIT:
                    break
        return found

    def _cards(self, asins: set[str]) -> dict[str, dict]:
        """Display fields for every product a response mentions."""
        return {asin: display.card(self.products, asin) for asin in asins}


def _asins(turns: list[dict]) -> set[str]:
    """Every product id anywhere in a turn list, slate or explanation."""
    found: set[str] = set()
    for turn in turns:
        found.update(turn.get("slate") or ())
        for item in turn.get("recommendations") or ():
            found.add(item["parent_asin"])
        ranking = (turn.get("explain") or {}).get("ranking") or {}
        for slot in ranking.get("slots") or ():
            found.add(slot["asin"])
        for slot in ranking.get("band") or ():
            found.add(slot["asin"])
        for slot in ranking.get("dropped_shown") or ():
            found.add(slot["asin"])
        goal = ranking.get("goal")
        if goal:
            found.add(goal["asin"])
    return found


class Handler(http.server.BaseHTTPRequestHandler):
    """Routes the handful of endpoints the page calls."""

    protocol_version = "HTTP/1.1"
    playground: Playground

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                return self._static("index.html")
            if route.startswith("/static/"):
                return self._static(route[len("/static/"):])
            if route == "/api/samples":
                return self._json(self.playground.sample_list())
            if route == "/api/product":
                asin = (query.get("asin") or [""])[0]
                return self._json(
                    display.card(self.playground.products, asin)
                )
            if route == "/api/search":
                term = (query.get("q") or [""])[0]
                return self._json(self.playground.search(term))
            return self._fail(404, "no such route")
        except Exception as error:  # Isolation point: one bad request only.
            return self._fail(500, f"{type(error).__name__}: {error}")

    def do_POST(self) -> None:
        route = urllib.parse.urlparse(self.path).path
        try:
            body = self._body()
            if route == "/api/replay":
                return self._json(
                    self.playground.replay(str(body.get("sample_id", "")))
                )
            if route == "/api/chat/open":
                return self._json(self.playground.open_live(
                    body.get("profile"), body.get("goal") or None
                ))
            if route == "/api/chat/send":
                return self._json(
                    self.playground.send_live(str(body.get("text", "")))
                )
            return self._fail(404, "no such route")
        except KeyError as error:
            return self._fail(404, f"not found: {error}")
        except Exception as error:  # Isolation point: one bad request only.
            return self._fail(500, f"{type(error).__name__}: {error}")

    def log_message(self, fmt: str, *args) -> None:
        """Quieter than the default, which prints every static asset."""
        if "/api/" in str(args[0] if args else ""):
            super().log_message(fmt, *args)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def _json(self, payload: object) -> None:
        self._send(200, "application/json", json.dumps(payload).encode())

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if not path.is_file() or STATIC.resolve() not in path.parents:
            return self._fail(404, "no such file")
        kind = mimetypes.guess_type(path.name)[0] or "text/plain"
        self._send(200, kind, path.read_bytes())

    def _fail(self, status: int, why: str) -> None:
        self._send(status, "application/json", json.dumps({
            "error": why
        }).encode())

    def _send(self, status: int, kind: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Builds the agent, then serves until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    print("building the agent and the catalog...", flush=True)
    Handler.playground = Playground(args.catalog, args.dataset)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"playground on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
