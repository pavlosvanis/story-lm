"""A small local-browser showcase for the trained StoryLM model.

Run with storylm gui. The interface needs only Python's standard library
and the project's existing inference dependencies.
"""

import json
import secrets
import threading
import webbrowser
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StoryLM</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center;
  background: #c0c0c0; color: #202020; font: 14px Arial, sans-serif; }
main { width: min(900px, calc(100vw - 32px)); height: min(650px, calc(100dvh - 32px));
  min-height: 300px; display: grid; grid-template-rows: auto auto minmax(0, 1fr) auto;
  gap: 10px; padding: 10px; background: #c0c0c0;
  border: 3px solid; border-color: #fff #777 #777 #fff; }
header { background: #000080; color: #fff; padding: 12px; font-size: 18px; font-weight: bold; }
nav, footer { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
button { padding: 8px 14px; background: #c0c0c0; color: #111; font: inherit;
  border: 2px solid; border-color: #fff #777 #777 #fff; cursor: pointer; }
button:active:not(:disabled) { border-color: #777 #fff #fff #777; }
button:disabled { color: #777; cursor: default; }
button:focus-visible, input:focus-visible { outline: 2px solid #000080; outline-offset: 2px; }
label { margin-left: auto; display: flex; align-items: center; gap: 6px; }
input { width: 150px; padding: 6px; background: #fffdf5; font: inherit; border: 1px solid #777; }
textarea { width: 100%; height: 100%; min-height: 0; resize: none; padding: 28px 32px;
  background: #fffdf5; color: #202020; font: 19px/1.55 "Courier New", monospace;
  border: 3px solid; border-color: #777 #fff #fff #777; outline: none; }
textarea:focus { box-shadow: inset 0 0 0 1px #000080; }
#status { flex: 1; min-width: 150px; padding: 8px; overflow-wrap: anywhere;
  border: 1px solid; border-color: #777 #fff #fff #777; }
@media (max-width: 540px) {
  main { width: calc(100vw - 16px); height: calc(100dvh - 16px); }
  textarea { padding: 16px; font-size: 17px; }
  label { margin-left: 0; }
}
</style>
</head>
<body>
<main>
<header>StoryLM — Story writer</header>
<nav aria-label="Story actions">
  <button id="new" type="button">New story</button>
  <button id="save" type="button">Save</button>
  <label>Filename <input id="filename" value="story" maxlength="80"></label>
</nav>
<textarea id="editor" aria-label="Story" spellcheck="false">Once upon a time</textarea>
<footer>
  <span id="status" role="status" aria-live="polite">Loading the model…</span>
  <button id="instant" type="button" disabled>Show instantly</button>
  <button id="continue" type="button" disabled>Continue the story</button>
</footer>
</main>
<script>
"use strict";
const editor = document.getElementById("editor");
const status = document.getElementById("status");
const newButton = document.getElementById("new");
const saveButton = document.getElementById("save");
const continueButton = document.getElementById("continue");
const instantButton = document.getElementById("instant");
const filename = document.getElementById("filename");
const initialText = editor.value;
let ready = false, loadFailed = false, busy = false, instant = false, wake = null;

function controls() {
  editor.readOnly = busy;
  newButton.disabled = busy;
  saveButton.disabled = busy;
  continueButton.disabled = busy || (!ready && !loadFailed);
  continueButton.textContent = loadFailed ? "Retry model" : busy ? "Writing…" : "Continue the story";
  instantButton.disabled = !busy || instant;
}
async function request(path, data) {
  const options = data === undefined ? {cache: "no-store"} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)
  };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request failed.");
  return result;
}
async function checkModel() {
  try {
    const model = await request("status");
    ready = model.ready;
    loadFailed = Boolean(model.error);
    status.textContent = model.error || (ready ? "Ready to write" : "Loading the model…");
    controls();
    if (!ready && !loadFailed) setTimeout(checkModel, 500);
  } catch (error) {
    ready = false;
    loadFailed = true;
    status.textContent = "Connection lost. Keep the terminal open, then select Retry model.";
    controls();
  }
}
function pause(milliseconds) {
  return new Promise(resolve => {
    const timer = setTimeout(() => { wake = null; resolve(); }, milliseconds);
    wake = () => { clearTimeout(timer); wake = null; resolve(); };
  });
}
async function reveal(prompt, completion) {
  for (const character of completion) {
    if (instant) { editor.value = prompt + completion; break; }
    editor.value += character;
    editor.scrollTop = editor.scrollHeight;
    await pause(".!?\n".includes(character) ? 150 : ",;:".includes(character) ? 65 : 22);
  }
  editor.scrollTop = editor.scrollHeight;
}
async function continueStory() {
  if (busy) return;
  if (!ready) {
    loadFailed = false;
    controls();
    status.textContent = "Loading the model…";
    try { await request("retry", {}); await checkModel(); }
    catch (error) { loadFailed = true; status.textContent = error.message; controls(); }
    return;
  }
  const prompt = editor.value;
  if (!prompt.trim()) { status.textContent = "Write a beginning first."; editor.focus(); return; }
  busy = true;
  instant = false;
  controls();
  status.textContent = "Writing…";
  try {
    const result = await request("generate", {prompt});
    await reveal(prompt, result.completion);
    status.textContent = result.completion ? "Ready to write" : "The model ended the story. Edit or try again.";
  } catch (error) {
    status.textContent = "Could not continue: " + error.message;
  } finally {
    busy = false;
    controls();
    if (document.activeElement === continueButton || document.activeElement === instantButton) editor.focus();
  }
}
function showInstantly() {
  if (!busy) return;
  instant = true;
  if (wake) wake();
  controls();
  status.textContent = "Writing… The result will appear all at once.";
}
function newStory() {
  if (busy) return;
  if (editor.value.trim() && editor.value !== initialText && !confirm("Clear the current story? Save it first if needed.")) return;
  editor.value = "";
  status.textContent = ready ? "Ready to write" : "The model is not ready yet.";
  editor.focus();
}
function saveStory() {
  if (busy) return;
  let name = filename.value.trim().replace(/[\\/:*?"<>|]/g, "_") || "story";
  name = name.replace(/(?:\.txt)+$/i, "") || "story";
  filename.value = name;
  const downloadName = name + ".txt";
  const url = URL.createObjectURL(new Blob([editor.value], {type: "text/plain;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = downloadName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  status.textContent = "Download requested: " + downloadName + ". Check your browser's downloads.";
}
newButton.addEventListener("click", newStory);
saveButton.addEventListener("click", saveStory);
continueButton.addEventListener("click", continueStory);
instantButton.addEventListener("click", showInstantly);
document.addEventListener("keydown", event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveStory();
  }
});
window.addEventListener("beforeunload", event => {
  if (busy || (editor.value.trim() && editor.value !== initialText)) {
    event.preventDefault();
    event.returnValue = "";
  }
});
controls();
checkModel();
</script>
</body>
</html>
"""


class _Model:
    """Load once and serialize calls to the existing generator."""

    def __init__(self, artifacts_dir: Path, device: str) -> None:
        self.artifacts_dir = artifacts_dir
        self.device = device
        self.generator = None
        self.error = ""
        self.ready = threading.Event()
        self.work = threading.Lock()

    def start_loading(self) -> None:
        """Retry initialization without blocking the browser or request thread."""
        if not self.work.acquire(blocking=False):
            raise RuntimeError("The model is already working. Please wait.")
        self.ready.clear()
        self.error = ""
        try:
            threading.Thread(target=self._load, daemon=True).start()
        except RuntimeError as error:
            self.error = str(error)
            self.work.release()
            self.ready.set()

    def _load(self) -> None:
        try:
            from storylm.inference.generator import StoryGenerator

            self.generator = StoryGenerator.from_artifacts(self.artifacts_dir, device=self.device)
        except Exception as error:
            self.error = f"Could not load the model: {error}"
        finally:
            self.work.release()
            self.ready.set()

    def generate(self, prompt: str) -> str:
        """Generate in an HTTP worker; permit only one inference call at a time."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Write a beginning first.")
        if not self.ready.is_set() or self.generator is None:
            raise RuntimeError(self.error or "The model is still loading.")
        if not self.work.acquire(blocking=False):
            raise RuntimeError("The model is already writing. Please wait.")
        try:
            return self.generator.generate(prompt, seed=secrets.randbits(32)).completion
        finally:
            self.work.release()


class _Handler(BaseHTTPRequestHandler):
    """Serve only the embedded interface and its local model endpoints."""

    timeout = 15

    def _route(self) -> str | None:
        if self.headers.get("Host") != self.server.host:
            self._reply({"error": "Invalid host."}, status=403)
            return None
        path = self.path.partition("?")[0]
        if not path.startswith(self.server.prefix):
            self._reply({"error": "Not found."}, status=404)
            return None
        return path[len(self.server.prefix) :]

    def _reply(self, body: dict | bytes, *, status: int = 200) -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        content_type = "text/html; charset=utf-8" if isinstance(body, bytes) else "application/json; charset=utf-8"
        with suppress(OSError):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

    def do_GET(self) -> None:
        """Return the interface or model initialization status."""
        route = self._route()
        if route is None:
            return
        if route == "":
            self._reply(_PAGE.encode("utf-8"))
        elif route == "status":
            model = self.server.model
            finished = model.ready.is_set()
            self._reply({"ready": finished and model.generator is not None, "error": model.error if finished else ""})
        else:
            self._reply({"error": "Not found."}, status=404)

    def do_POST(self) -> None:
        """Accept JSON generation and load-retry requests from this interface."""
        route = self._route()
        if route is None:
            return
        if self.headers.get("Origin", self.server.origin) != self.server.origin:
            self._reply({"error": "Invalid origin."}, status=403)
            return
        if route not in ("generate", "retry"):
            self._reply({"error": "Not found."}, status=404)
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON request.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 512_000:
                raise ValueError("The request is empty or too large.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            if route == "retry":
                if self.server.model.generator is None:
                    self.server.model.start_loading()
                self._reply({"ok": True})
            else:
                completion = self.server.model.generate(data.get("prompt"))
                self._reply({"completion": completion})
        except ValueError as error:
            self._reply({"error": str(error)}, status=400)
        except RuntimeError as error:
            self._reply({"error": str(error)}, status=409)
        except Exception as error:
            self._reply({"error": str(error) or type(error).__name__}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        """Keep normal requests out of the showcase terminal output."""


class _Server(ThreadingHTTPServer):
    """Expose the showcase only on loopback under a fresh private URL."""

    def __init__(self, artifacts_dir: Path, device: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.host = f"127.0.0.1:{self.server_port}"
        self.origin = f"http://{self.host}"
        self.prefix = f"/{secrets.token_urlsafe(24)}/"
        self.model = _Model(artifacts_dir, device)


def launch_gui(artifacts_dir: str | Path = "artifacts", *, device: str = "auto") -> None:
    """Open the local writer; stop its server with Ctrl+C in the terminal."""
    artifacts_dir = Path(artifacts_dir).expanduser().resolve()
    with _Server(artifacts_dir, device) as server:
        server.model.start_loading()
        url = f"{server.origin}{server.prefix}"
        print(f"StoryLM: {url}\nKeep this terminal open. Press Ctrl+C to stop.")
        with suppress(webbrowser.Error, OSError):
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
