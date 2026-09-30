"""Tests for the local-browser GUI, without loading model artifacts.

HTTP tests exercise the real loopback server. Optional Node tests run the
embedded script with a small DOM double; they do not replace browser testing.
"""

import json
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from storylm import gui


@pytest.fixture
def model():
    return gui._Model(Path("artifacts"), "cpu")


@pytest.fixture
def factory(monkeypatch):
    """Keep model-loading tests independent of Torch and artifact files."""
    module = ModuleType("storylm.inference.generator")
    factory = Mock(return_value=Mock())
    module.StoryGenerator = SimpleNamespace(from_artifacts=factory)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return factory


def loaded_model(model, completion=" A new paragraph.\n\nΚαλημέρα 🌍"):
    model.generator = Mock()
    model.generator.generate.return_value = SimpleNamespace(completion=completion)
    model.ready.set()
    return model


def test_load_uses_requested_artifacts_and_device(model, factory):
    model.start_loading()

    assert model.ready.wait(3), "Model loading did not finish"
    factory.assert_called_once_with(Path("artifacts"), device="cpu")
    assert model.generator is factory.return_value
    assert model.error == ""
    assert not model.work.locked()


def test_failed_load_can_be_retried(model, factory):
    factory.side_effect = [OSError("missing artifacts"), factory.return_value]
    model.start_loading()

    assert model.ready.wait(3)
    assert model.generator is None
    assert "missing artifacts" in model.error
    assert not model.work.locked()
    with pytest.raises(RuntimeError, match="missing artifacts"):
        model.generate("Hello")

    model.start_loading()

    assert model.ready.wait(3)
    assert model.generator is factory.return_value
    assert model.error == ""
    assert factory.call_count == 2
    assert not model.work.locked()


def test_thread_start_failure_can_be_retried(model, factory, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(gui.threading, "Thread", Mock(side_effect=RuntimeError("no threads")))
        model.start_loading()

    assert model.ready.is_set()
    assert model.error == "no threads"
    assert not model.work.locked()

    model.start_loading()
    assert model.ready.wait(3)
    assert model.generator is factory.return_value
    assert model.error == ""


def test_duplicate_loading_does_not_start_a_second_worker(model, factory):
    entered = threading.Event()
    release = threading.Event()

    def load(*args, **kwargs):
        entered.set()
        assert release.wait(5), "Test did not release the loader"
        return SimpleNamespace()

    factory.side_effect = load
    model.start_loading()
    try:
        assert entered.wait(3)
        assert not model.ready.is_set()
        with pytest.raises(RuntimeError, match="already working"):
            model.start_loading()
        factory.assert_called_once()
    finally:
        release.set()
        assert model.ready.wait(3)

    assert not model.work.locked()


@pytest.mark.parametrize("prompt", [None, 42, {}, [], "", " \n\t"])
def test_invalid_prompt_never_calls_generator(model, prompt):
    loaded_model(model)
    with pytest.raises(ValueError, match="beginning"):
        model.generate(prompt)
    model.generator.generate.assert_not_called()
    assert not model.work.locked()


def test_generation_rejects_loading_or_failed_model(model):
    with pytest.raises(RuntimeError, match="still loading"):
        model.generate("Hello")

    model.error = "Could not load the model"
    model.ready.set()
    with pytest.raises(RuntimeError, match="Could not load"):
        model.generate("Hello")


def test_generation_preserves_prompt_completion_and_decoder_defaults(model, monkeypatch):
    completion = " the end.\n\nΚαλημέρα 🌍"
    loaded_model(model, completion)
    monkeypatch.setattr(gui.secrets, "randbits", Mock(return_value=123))
    prompt = "Once upon a time\n"

    assert model.generate(prompt) == completion
    model.generator.generate.assert_called_once_with(prompt, seed=123)
    gui.secrets.randbits.assert_called_once_with(32)
    assert not model.work.locked()


def test_generation_failure_releases_lock_for_next_request(model):
    loaded_model(model)
    model.generator.generate.side_effect = [
        OSError("inference failed"),
        SimpleNamespace(completion=" recovered"),
    ]

    with pytest.raises(OSError, match="inference failed"):
        model.generate("Hello")
    assert not model.work.locked()
    assert model.generate("Hello") == " recovered"


@pytest.fixture
def server():
    with gui._Server(Path("artifacts"), "cpu") as server:
        worker = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        worker.start()
        try:
            yield server
        finally:
            server.shutdown()
            worker.join(timeout=3)
            assert not worker.is_alive(), "HTTP server did not stop"


def request(server, route="", *, method="GET", data=None, body=None, headers=None, path=None):
    """Use a real socket, with explicit timeouts and guaranteed cleanup."""
    request_headers = dict(headers or {})
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, server.prefix + route if path is None else path, body, request_headers)
        response = connection.getresponse()
        payload = response.read()
        response_headers = dict(response.getheaders())
        result = (
            json.loads(payload)
            if response_headers["Content-Type"].startswith("application/json")
            else payload.decode("utf-8")
        )
        return response.status, response_headers, result
    finally:
        connection.close()


def test_server_binds_loopback_with_distinct_private_urls(server):
    assert server.server_address[0] == "127.0.0.1"
    assert server.origin == f"http://127.0.0.1:{server.server_port}"
    assert server.prefix.startswith("/") and server.prefix.endswith("/")
    with gui._Server(Path("artifacts"), "cpu") as other:
        assert other.prefix != server.prefix


def test_page_and_response_headers(server):
    status, headers, page = request(server)

    assert status == 200
    assert page == gui._PAGE
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert int(headers["Content-Length"]) == len(page.encode("utf-8"))
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"


def test_status_tracks_loading_failure_and_success(server):
    server.model.error = "previous error"
    assert request(server, "status?fresh=1")[2] == {"ready": False, "error": ""}

    server.model.ready.set()
    assert request(server, "status")[2] == {"ready": False, "error": "previous error"}

    loaded_model(server.model)
    server.model.error = ""
    assert request(server, "status")[2] == {"ready": True, "error": ""}


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_invalid_host_is_rejected(server, method):
    status, _, result = request(
        server, "status", method=method, headers={"Host": "example.com"}
    )
    assert status == 403
    assert "host" in result["error"].lower()


@pytest.mark.parametrize("path", ["/", "/status", "/wrong-token/generate"])
def test_paths_outside_private_url_are_rejected(server, path):
    assert request(server, path=path)[0] == 404


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_unknown_endpoint_is_rejected(server, method):
    assert request(server, "unknown", method=method, data={} if method == "POST" else None)[0] == 404


def test_cross_origin_generation_is_rejected_before_inference(server):
    loaded_model(server.model)
    status, _, result = request(
        server,
        "generate",
        method="POST",
        data={"prompt": "Hello"},
        headers={"Origin": "https://example.com"},
    )
    assert status == 403
    assert "origin" in result["error"].lower()
    server.model.generator.generate.assert_not_called()


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (b'{}', {"Content-Type": "text/plain"}),
        (b'{', {"Content-Type": "application/json"}),
        (b'[]', {"Content-Type": "application/json"}),
        (b'null', {"Content-Type": "application/json"}),
        (b'', {"Content-Type": "application/json"}),
        (b'{}', {"Content-Type": "application/json", "Content-Length": "-1"}),
        (b'{}', {"Content-Type": "application/json", "Content-Length": "invalid"}),
        (b'{}', {"Content-Type": "application/json", "Content-Length": "512001"}),
    ],
)
def test_invalid_json_requests_are_rejected(server, body, headers):
    loaded_model(server.model)
    status, _, result = request(server, "generate", method="POST", body=body, headers=headers)
    assert status == 400
    assert result["error"]
    server.model.generator.generate.assert_not_called()


@pytest.mark.parametrize("data", [{}, {"prompt": ""}, {"prompt": 123}, {"prompt": " \n"}])
def test_http_invalid_prompts_are_rejected(server, data):
    loaded_model(server.model)
    assert request(server, "generate", method="POST", data=data)[0] == 400
    server.model.generator.generate.assert_not_called()


def test_http_generation_and_repeated_requests_preserve_unicode(server):
    completion = "\n\nΚαλημέρα 🌍"
    loaded_model(server.model, completion)
    for prompt in ("Hello", "Hello" + completion, "Hello" + completion * 2):
        status, headers, result = request(
            server, "generate", method="POST", data={"prompt": prompt},
            headers={"Origin": server.origin},
        )
        assert status == 200
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert result == {"completion": completion}
    assert server.model.generator.generate.call_count == 3


def test_http_not_ready_and_inference_failure_allow_recovery(server):
    assert request(server, "generate", method="POST", data={"prompt": "Hello"})[0] == 409
    loaded_model(server.model)
    server.model.generator.generate.side_effect = [
        OSError("failed inference"),
        SimpleNamespace(completion=""),
    ]

    status, _, result = request(server, "generate", method="POST", data={"prompt": "Hello"})
    assert status == 500
    assert result == {"error": "failed inference"}
    assert request(server, "generate", method="POST", data={"prompt": "Hello"})[2] == {"completion": ""}


def test_status_stays_responsive_and_parallel_generation_is_rejected(server):
    loaded_model(server.model)
    entered = threading.Event()
    release = threading.Event()

    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(5), "Test did not release inference"
        return SimpleNamespace(completion=" finished")

    server.model.generator.generate.side_effect = generate
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(request, server, "generate", method="POST", data={"prompt": "Hello"})
        try:
            assert entered.wait(3)
            assert request(server, "status")[2] == {"ready": True, "error": ""}
            assert request(server, "generate", method="POST", data={"prompt": "Again"})[0] == 409
            server.model.generator.generate.assert_called_once()
        finally:
            release.set()
        assert first.result(timeout=3)[2] == {"completion": " finished"}

    assert request(server, "generate", method="POST", data={"prompt": "Again"})[0] == 200


def test_http_retry_loads_failed_model_and_does_not_reload_ready_model(server, factory):
    server.model.error = "failed load"
    server.model.ready.set()
    assert request(server, "retry", method="POST", data={})[2] == {"ok": True}
    assert server.model.ready.wait(3)
    assert request(server, "status")[2] == {"ready": True, "error": ""}
    assert request(server, "retry", method="POST", data={})[2] == {"ok": True}
    factory.assert_called_once_with(Path("artifacts"), device="cpu")


@pytest.mark.parametrize("browser_error", [OSError, gui.webbrowser.Error])
def test_launch_preserves_cli_arguments_and_closes_server_on_interrupt(monkeypatch, tmp_path, capsys, browser_error):
    server = Mock()
    server.origin = "http://127.0.0.1:12345"
    server.prefix = "/private/"
    server.serve_forever.side_effect = KeyboardInterrupt
    context = Mock()
    context.__enter__ = Mock(return_value=server)
    context.__exit__ = Mock(return_value=False)
    constructor = Mock(return_value=context)
    monkeypatch.setattr(gui, "_Server", constructor)
    monkeypatch.setattr(gui.webbrowser, "open", Mock(side_effect=browser_error("no browser")))

    gui.launch_gui(str(tmp_path), device="cpu")

    constructor.assert_called_once_with(tmp_path.resolve(), "cpu")
    server.model.start_loading.assert_called_once_with()
    gui.webbrowser.open.assert_called_once_with(server.origin + server.prefix)
    server.serve_forever.assert_called_once_with()
    context.__exit__.assert_called_once()
    assert server.origin + server.prefix in capsys.readouterr().out


# Run the real embedded script. DOM doubles let us test state transitions and
# exported text without a display server, third-party JS packages, or inference.
_BROWSER_HARNESS = r"""
const assert = require("node:assert/strict");
const vm = require("node:vm");
const script = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const scenario = process.argv[1];
const completion = " the end.\n\nΚαλημέρα 🌍";
const elements = new Map();
function element(id) {
  return {id, value: id === "editor" ? "Once upon a time" : id === "filename" ? "story.txt" : "",
    disabled: false, readOnly: false, textContent: "", scrollTop: 0, scrollHeight: 100,
    handlers: {}, addEventListener(kind, handler) { this.handlers[kind] = handler; },
    focus() { document.activeElement = this; }};
}
for (const id of ["editor", "status", "new", "save", "continue", "instant", "filename"]) {
  elements.set(id, element(id));
}
const downloads = [], timers = new Map(), requests = [], answers = [];
let nextTimer = 0, confirmation = true;
const document = {
  activeElement: null, handlers: {},
  getElementById(id) { return elements.get(id); },
  addEventListener(kind, handler) { this.handlers[kind] = handler; },
  body: {appendChild() {}},
  createElement() {
    return {click() { downloads.push({name: this.download, blob: blobs.get(this.href)}); }, remove() {}};
  }
};
const window = {handlers: {}, addEventListener(kind, handler) { this.handlers[kind] = handler; }};
const blobs = new Map();
let nextUrl = 0;
const context = vm.createContext({
  document, window, Blob, confirm: () => confirmation,
  URL: {createObjectURL(blob) { const url = "blob:" + ++nextUrl; blobs.set(url, blob); return url; },
    revokeObjectURL(url) { blobs.delete(url); }},
  setTimeout(fn, delay) { const id = ++nextTimer; timers.set(id, {fn, delay}); return id; },
  clearTimeout(id) { timers.delete(id); },
  async fetch(path, options) {
    requests.push({path, options});
    assert.ok(answers.length, "Unexpected request: " + path);
    const answer = answers.shift();
    if (typeof answer === "function") return await answer(path, options);
    return response(answer);
  }
});
function response(body, ok = true) { return {ok, json: async () => body}; }
function click(id) {
  const target = elements.get(id);
  if (target.disabled) return Promise.resolve();
  document.activeElement = target;
  return Promise.resolve(target.handlers.click());
}
async function flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); }
function runAnimationTimer() {
  for (const [id, timer] of timers) {
    if (timer.delay < 500) { timers.delete(id); timer.fn(); return true; }
  }
  return false;
}
async function finishAnimation() {
  for (let i = 0; i < 1000; i++) {
    await flush();
    if (!elements.get("editor").readOnly) return;
    assert.ok(runAnimationTimer(), "Animation stopped without restoring editing");
  }
  assert.fail("Animation never finished");
}
const editor = elements.get("editor");
answers.push({ready: scenario !== "retry", error: scenario === "retry" ? "failed load" : ""});
vm.runInContext(script, context);
(async () => {
  await flush();
  if (scenario === "cycles") {
    for (let i = 0; i < 3; i++) {
      const before = editor.value;
      answers.push({completion});
      const writing = click("continue");
      await finishAnimation();
      await writing;
      assert.equal(editor.value, before + completion);
      assert.equal(elements.get("save").disabled, false);
      assert.equal(elements.get("continue").disabled, false);
      await click("save");
      assert.equal(await downloads[i].blob.text(), editor.value);
      assert.equal(downloads[i].name, "story.txt");
      assert.equal(elements.get("filename").value, "story");
    }
  } else if (scenario === "instant-before") {
    let resolveResponse;
    answers.push(() => new Promise(resolve => { resolveResponse = resolve; }));
    const before = editor.value;
    const writing = click("continue");
    await flush();
    assert.equal(editor.readOnly, true);
    await click("continue");
    await click("save");
    await click("new");
    assert.equal(requests.filter(item => item.path === "generate").length, 1);
    assert.equal(downloads.length, 0);
    assert.equal(editor.value, before);
    await click("instant");
    resolveResponse(response({completion}));
    await writing;
    assert.equal(editor.value, before + completion);
    assert.equal(editor.readOnly, false);
    assert.equal(elements.get("continue").disabled, false);
  } else if (scenario === "instant-during") {
    const before = editor.value;
    answers.push({completion});
    const writing = click("continue");
    await flush();
    assert.equal(editor.value, before + completion[0]);
    await click("instant");
    await writing;
    assert.equal(editor.value, before + completion);
    assert.equal(editor.readOnly, false);
  } else if (scenario === "failure") {
    const before = editor.value;
    answers.push(() => response({error: "inference failed"}, false));
    await click("continue");
    assert.equal(editor.value, before);
    assert.equal(editor.readOnly, false);
    assert.equal(elements.get("continue").disabled, false);
    assert.match(elements.get("status").textContent, /inference failed/);
    answers.push({completion});
    const writing = click("continue");
    await flush();
    await click("instant");
    await writing;
    assert.equal(editor.value, before + completion);
  } else if (scenario === "retry") {
    assert.equal(elements.get("continue").textContent, "Retry model");
    assert.equal(elements.get("continue").disabled, false);
    answers.push({ok: true}, {ready: true, error: ""});
    await click("continue");
    assert.equal(elements.get("continue").textContent, "Continue the story");
    assert.deepEqual(requests.map(item => item.path), ["status", "retry", "status"]);
  } else if (scenario === "document") {
    editor.value = "A changed story";
    confirmation = false;
    await click("new");
    assert.equal(editor.value, "A changed story");
    confirmation = true;
    await click("new");
    assert.equal(editor.value, "");
    await click("continue");
    assert.match(elements.get("status").textContent, /beginning/);
    assert.equal(requests.length, 1);
    editor.value = "Καλημέρα\n\n🌍";
    elements.get("filename").value = " bad/name ";
    let prevented = false;
    document.handlers.keydown({ctrlKey: true, key: "s", preventDefault() { prevented = true; }});
    assert.equal(prevented, true);
    assert.equal(downloads[0].name, "bad_name.txt");
    assert.equal(elements.get("filename").value, "bad_name");
    assert.equal(await downloads[0].blob.text(), editor.value);
    for (const [input, base] of [["My story", "My story"], ["My story.txt", "My story"],
      ["My story.TXT.txt", "My story"], ["  ", "story"]]) {
      elements.get("filename").value = input;
      await click("save");
      assert.equal(elements.get("filename").value, base);
      assert.equal(downloads[downloads.length - 1].name, base + ".txt");
      assert.equal(await downloads[downloads.length - 1].blob.text(), editor.value);
    }
    assert.equal(elements.get("continue").disabled, false);
    let warned = false;
    window.handlers.beforeunload({preventDefault() { warned = true; }});
    assert.equal(warned, true);
    assert.equal(editor.value, "Καλημέρα\n\n🌍");
  } else if (scenario === "empty") {
    const before = editor.value;
    answers.push({completion: ""});
    await click("continue");
    assert.equal(editor.value, before);
    assert.equal(editor.readOnly, false);
    assert.equal(elements.get("continue").disabled, false);
    assert.match(elements.get("status").textContent, /ended/);
  } else {
    assert.fail("Unknown scenario: " + scenario);
  }
  assert.equal(answers.length, 0, "Not all expected requests were made");
  console.log("PASS " + scenario);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize(
    "scenario",
    ["cycles", "instant-before", "instant-during", "failure", "retry", "document", "empty"],
)
def test_browser_controls_and_downloads(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed only for the embedded-script tests")
    script = gui._PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run(
        [node, "-e", _BROWSER_HARNESS, scenario],
        input=json.dumps(script),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {scenario}" in result.stdout, "Browser scenario did not reach its final assertions"
