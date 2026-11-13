"""
Tests for the ASGI ``http.response.zerocopysend`` extension.

``os.sendfile`` does not exist on Windows, so the production capability
is correctly hidden there.  To exercise the full transfer logic on every
platform these tests run a real reactor with real loopback TCP sockets
(real bytes traverse the kernel stack and are asserted on the receiver)
and inject a scripted sendfile backend that reproduces the documented
``os.sendfile`` semantics -- including forced short sends, EAGAIN and
the two file position rules.  The fake is a test-only syscall double;
the production capability gate never treats an ordinary read/write path
as zero copy.
"""

import asyncio
import errno
import io
import os
import socket
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from unittest import mock

# isort: split
# Importing daphne.server installs the asyncio reactor; it must run before
# anything that imports twisted.internet.reactor directly (which would
# otherwise install the wrong default reactor).
import daphne.zerocopy as zerocopy
from daphne.http_protocol import ZEROCOPY_EXTENSION, HTTPFactory
from daphne.server import Server

# isort: split
from twisted.internet import defer, endpoints, reactor
from twisted.internet.tcp import Connection as TCPConnection


def run_in_reactor(callable_, *args, timeout=15, **kwargs):
    """
    Run ``callable_`` on the reactor thread and return its result.
    Deferreds are resolved before returning; failures are re-raised in
    the calling thread.
    """
    box = {}
    done = threading.Event()

    def _run():
        try:
            result = callable_(*args, **kwargs)
        except BaseException as exc:  # re-raised in the caller thread
            box["error"] = exc
            done.set()
            return
        if isinstance(result, defer.Deferred):

            def _ok(value):
                box["value"] = value
                done.set()

            def _err(failure):
                box["failure"] = failure
                done.set()

            result.addCallbacks(_ok, _err)
        else:
            box["value"] = result
            done.set()

    reactor.callFromThread(_run)
    if not done.wait(timeout):
        raise TimeoutError("reactor work did not finish within %ss" % timeout)
    if "error" in box:
        raise box["error"]
    if "failure" in box:
        box["failure"].raiseException()
    return box.get("value")


class RoutingApplication:
    """ASGI application that delegates to the handler installed per test."""

    def __init__(self):
        self.handler = None

    async def __call__(self, scope, receive, send):
        message = await receive()
        while message.get("more_body"):
            message = await receive()
        await self.handler(scope, receive, send)


class FakeSendfile:
    """
    Test double for ``os.sendfile(out_fd, in_fd, offset, count)``.

    Reproduces the publicly documented semantics:

    * ``offset is None``: bytes are read from the descriptor's current
      position, which is advanced by the number of bytes transferred;
    * an explicit integer ``offset``: bytes are read from that offset and
      the descriptor's own position is left untouched.

    ``short_cap`` limits the bytes moved per call (forced short sends);
    ``eagain_calls`` makes the first N calls raise EAGAIN, after which
    transfers resume normally.
    """

    def __init__(self, server, short_cap=None, eagain_calls=0):
        self.server = server
        self.short_cap = short_cap
        self.eagain_calls = eagain_calls
        self.call_log = []
        self.total_sent = 0

    def _resolve_out_socket(self, out_fd):
        # server.connections may still hold protocols whose transports
        # have already lost their sockets; skip those and match the live
        # descriptor exactly.
        for protocol in list(self.server.connections.keys()):
            transport = getattr(protocol, "transport", None)
            get_handle = getattr(transport, "getHandle", None)
            if get_handle is None:
                continue
            try:
                sock = get_handle()
            except Exception:
                continue
            if not isinstance(sock, socket.socket):
                continue
            try:
                fileno = sock.fileno()
            except OSError:
                continue
            if fileno == out_fd:
                return sock
        raise AssertionError("no server socket found for out fd %r" % (out_fd,))

    def __call__(self, out_fd, in_fd, offset, count):
        assert isinstance(count, int) and count > 0
        if self.eagain_calls > 0:
            self.eagain_calls -= 1
            raise BlockingIOError(errno.EAGAIN, "resource temporarily unavailable")
        want = count if self.short_cap is None else min(count, self.short_cap)
        if offset is None:
            # Implicit offset: consume and advance the file position.
            data = os.read(in_fd, want)
        else:
            # Explicit offset: pread-style, restore the position afterwards.
            saved = os.lseek(in_fd, 0, os.SEEK_CUR)
            os.lseek(in_fd, offset, os.SEEK_SET)
            data = os.read(in_fd, want)
            os.lseek(in_fd, saved, os.SEEK_SET)
        out_socket = self._resolve_out_socket(out_fd)
        # The socket is non-blocking; a natural EAGAIN here is allowed to
        # propagate, exactly like the real syscall would report it.
        sent = out_socket.send(data)
        self.call_log.append((offset, count, sent))
        self.total_sent += sent
        return sent


class ReactorHarness:
    """Starts one in-process Daphne server driven by a background reactor."""

    def __init__(self):
        self.application = RoutingApplication()
        self.server = Server(
            application=self.application,
            endpoints=["tcp:port=0:interface=127.0.0.1"],
            signal_handlers=False,
        )
        # Server.run() normally creates this; we wire the factory manually.
        self.server.connections = {}
        self.factory = HTTPFactory(self.server)
        self.port = None

    def start(self):
        def _listen():
            endpoint = endpoints.serverFromString(
                reactor, "tcp:port=0:interface=127.0.0.1"
            )
            return endpoint.listen(self.factory)

        listener = run_in_reactor(_listen)
        self.port = listener.getHost().port
        return self

    def set_handler(self, handler):
        self.application.handler = handler

    def find_protocol(self, client_port):
        def _find():
            for protocol in list(self.server.connections.keys()):
                transport = getattr(protocol, "transport", None)
                if transport is None:
                    continue
                try:
                    if transport.getPeer().port == client_port:
                        return protocol
                except Exception:
                    continue
            return None

        return run_in_reactor(_find)


_reactor_ready = threading.Event()


def _reactor_thread():
    if reactor.running:
        _reactor_ready.set()
        return
    reactor.callWhenRunning(_reactor_ready.set)
    reactor.run(installSignalHandlers=False)


def _recv_more(sock, buf, needed):
    while len(buf) < needed:
        chunk = sock.recv(max(8192, needed - len(buf)))
        if not chunk:
            raise AssertionError("connection closed mid-response")
        buf += chunk
    return buf


def _dechunk(sock, initial):
    """Parse a chunked transfer encoded body from a raw socket."""
    buf = initial
    body = b""
    while True:
        while b"\r\n" not in buf:
            chunk = sock.recv(8192)
            if not chunk:
                raise AssertionError("connection closed inside chunk framing")
            buf += chunk
        size_line, _, buf = buf.partition(b"\r\n")
        chunk_size = int(size_line.split(b";", 1)[0], 16)
        if chunk_size == 0:
            # Swallow the trailing CRLF of the terminating chunk.
            while b"\r\n" not in buf:
                got = sock.recv(8192)
                if not got:
                    break
                buf += got
            return body
        frame_end = chunk_size + 2
        buf = _recv_more(sock, buf, frame_end)
        if buf[chunk_size:frame_end] != b"\r\n":
            raise AssertionError("malformed chunk framing")
        body += buf[:chunk_size]
        buf = buf[frame_end:]


def read_raw_response(sock, expect_body=True, timeout=10):
    """
    Read a full HTTP/1.1 response over a raw socket.

    Handles Content-Length framing, chunked transfer encoding and
    bodyless responses (HEAD / 204 / 304), where the headers alone are
    returned with an empty body.
    """
    sock.settimeout(timeout)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(8192)
        if not chunk:
            break
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    content_length = None
    chunked = False
    for line in head.split(b"\r\n"):
        lower = line.lower()
        if lower.startswith(b"content-length:"):
            content_length = int(line.split(b":", 1)[1].strip())
        elif lower.startswith(b"transfer-encoding:") and b"chunked" in lower:
            chunked = True
    if not expect_body:
        return head, b""
    if chunked:
        return head, _dechunk(sock, body)
    if content_length is not None:
        body = _recv_more(sock, body, content_length)
        return head, body[:content_length]
    # No framing information: read until the server closes the connection.
    try:
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            body += chunk
    except socket.timeout:
        pass
    return head, body


def open_client(harness):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect(("127.0.0.1", harness.port))
    return sock


def basic_request(sock, path="/"):
    sock.sendall(
        (
            "GET %s HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n" % path
        ).encode("ascii")
    )


class ZeroCopySendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threading.Thread(target=_reactor_thread, daemon=True).start()
        if not _reactor_ready.wait(10):
            raise RuntimeError("reactor did not start")
        cls.harness = ReactorHarness().start()
        cls._tempdir = tempfile.TemporaryDirectory()
        cls.tempdir = cls._tempdir.name

    @classmethod
    def tearDownClass(cls):
        cls._tempdir.cleanup()

    def setUp(self):
        # Distinct payload per test so cross-test contamination is visible.
        self.payload = bytes((i % 251 for i in range(4096)))
        fd, self.temp_path = tempfile.mkstemp(dir=self.tempdir)
        with os.fdopen(fd, "wb") as fh:
            fh.write(self.payload)

    def _handler(self, fn):
        """Wrap a handler body with the standard request drain."""

        async def handler(scope, receive, send):
            await fn(scope, receive, send)

        return handler

    def _run_request(self, handler, fake=None, path="/"):
        self.harness.set_handler(handler)
        sock = open_client(self.harness)
        client_port = sock.getsockname()[1]
        try:
            if fake is None:
                basic_request(sock, path)
                head, body = read_raw_response(sock)
            else:
                with mock.patch.object(zerocopy, "kernel_sendfile", fake):
                    basic_request(sock, path)
                    head, body = read_raw_response(sock)
        finally:
            sock.close()
        return client_port, head, body

    ### Capability policy

    def test_extension_declared_only_with_sendfile_transport(self):
        captured = {}
        done = threading.Event()

        async def handler(scope, receive, send):
            captured["scope"] = scope
            done.set()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})

        # Without a sendfile backend (the real situation on Windows) the
        # extension must not be advertised.
        _, _, _ = self._run_request(self._handler(handler))
        self.assertTrue(done.wait(10))
        self.assertNotIn("extensions", captured["scope"])

        # With a backend the real plaintext loopback TCP transport is
        # genuinely sendfile-capable, so the extension appears.
        done.clear()
        fake = FakeSendfile(self.harness.server)
        _, _, _ = self._run_request(self._handler(handler), fake=fake)
        self.assertTrue(done.wait(10))
        self.assertEqual(captured["scope"].get("extensions"), {ZEROCOPY_EXTENSION: {}})

    def test_capability_rejects_non_socket_transports(self):
        # TLS wrappers and H2 channels are not tcp.Connection instances.
        fake_transport = mock.Mock()
        fake_transport.getHandle.return_value = mock.sentinel.not_a_socket
        with mock.patch.object(zerocopy, "kernel_sendfile", lambda *a: 0):
            self.assertFalse(zerocopy.zerocopy_capable(fake_transport))
            self.assertFalse(zerocopy.zerocopy_capable(None))

        # A live plaintext loopback TCP transport is genuinely capable
        # once a backend exists.  Evaluate while the connection and its
        # socket descriptor are still live, inside the reactor thread.
        done = threading.Event()
        captured = {}

        async def handler(scope, receive, send):
            protocol = self._current_protocol(client_port)
            transport = protocol.transport if protocol is not None else None
            captured["transport"] = transport
            captured["capable"] = zerocopy.zerocopy_capable(transport)
            done.set()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})

        self.harness.set_handler(self._handler(handler))
        sock = open_client(self.harness)
        client_port = sock.getsockname()[1]
        try:
            with mock.patch.object(zerocopy, "kernel_sendfile", lambda *a: 0):
                basic_request(sock)
                read_raw_response(sock)
        finally:
            sock.close()
        self.assertTrue(done.wait(10))
        self.assertIsInstance(captured["transport"], TCPConnection)
        self.assertTrue(captured["capable"])

    def test_zerocopysend_refused_without_capability(self):
        # On a platform/transport without os.sendfile the message must be
        # rejected rather than silently downgraded to read/write.
        state = {}
        done = threading.Event()

        async def handler(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            with open(self.temp_path, "rb") as fh:
                try:
                    await send(
                        {
                            "type": "http.response.zerocopysend",
                            "file": fh,
                            "count": 10,
                        }
                    )
                except ValueError as exc:
                    state["error"] = exc
            await send({"type": "http.response.body", "body": b"fallback"})
            done.set()

        _, head, body = self._run_request(self._handler(handler))
        self.assertTrue(done.wait(10))
        self.assertIsInstance(state["error"], ValueError)
        self.assertEqual(body, b"fallback")
        self.assertNotIn(b"\x00", head)

    ### Argument validation (all rejected before any byte is written)

    def test_invalid_arguments_rejected_before_writes(self):
        bad_messages = [
            {"file": None},
            {"file": io.BytesIO(b"abc")},
            {"file": object()},
            {"file": open(self.temp_path, "rb"), "offset": -1},
            {"file": open(self.temp_path, "rb"), "count": -5},
            {"file": open(self.temp_path, "rb"), "offset": 1.5},
            {"file": open(self.temp_path, "rb"), "offset": True},
            {"file": open(self.temp_path, "rb"), "offset": 1 << 63},
            {"file": open(self.temp_path, "rb"), "count": 1 << 64},
        ]
        for message in bad_messages:
            with self.subTest(message=message):
                state = {}
                done = threading.Event()
                bad = dict(message)
                file_obj = bad.pop("file")

                async def handler(scope, receive, send, file_obj=file_obj, bad=bad):
                    await send({"type": "http.response.start", "status": 200})
                    try:
                        await send(
                            {
                                "type": "http.response.zerocopysend",
                                "file": file_obj,
                                **bad,
                            }
                        )
                    except (ValueError, OverflowError) as exc:
                        state["error"] = exc
                    finally:
                        if hasattr(file_obj, "close"):
                            file_obj.close()
                    await send({"type": "http.response.body", "body": b"safe"})
                    done.set()

                _, _, body = self._run_request(
                    self._handler(handler), fake=FakeSendfile(self.harness.server)
                )
                self.assertTrue(done.wait(10))
                self.assertIn("error", state)
                self.assertEqual(body, b"safe")

    def test_closed_file_descriptor_rejected(self):
        state = {}
        done = threading.Event()
        fh = open(self.temp_path, "rb")
        fh.close()

        async def handler(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            try:
                await send(
                    {"type": "http.response.zerocopysend", "file": fh, "count": 4}
                )
            except ValueError as exc:
                state["error"] = exc
            await send({"type": "http.response.body", "body": b"safe"})
            done.set()

        _, _, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        self.assertIsInstance(state["error"], ValueError)
        self.assertEqual(body, b"safe")

    ### Transfer semantics over real sockets

    def test_explicit_offset_and_count_keeps_file_position(self):
        state = {}
        done = threading.Event()
        offset, count = 100, 64

        async def handler(scope, receive, send):
            with open(self.temp_path, "rb") as fh:
                fh.seek(7)
                before = fh.tell()
                await send({"type": "http.response.start", "status": 200})
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": offset,
                        "count": count,
                    }
                )
                state["position_after"] = fh.tell()
                state["position_before"] = before
            done.set()

        _, _, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        range_end = offset + count
        self.assertEqual(body, self.payload[offset:range_end])
        # Explicit offset: the descriptor position must not move.
        self.assertEqual(state["position_before"], 7)
        self.assertEqual(state["position_after"], 7)

    def test_implicit_offset_uses_and_advances_current_position(self):
        state = {}
        done = threading.Event()
        start, count = 33, 128

        async def handler(scope, receive, send):
            with open(self.temp_path, "rb") as fh:
                fh.seek(start)
                await send({"type": "http.response.start", "status": 200})
                await send(
                    {"type": "http.response.zerocopysend", "file": fh, "count": count}
                )
                state["position_after"] = fh.tell()
            done.set()

        _, _, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        range_end = start + count
        self.assertEqual(body, self.payload[start:range_end])
        self.assertEqual(state["position_after"], start + count)

    def test_zero_count_sends_nothing_and_keeps_position(self):
        state = {}
        done = threading.Event()

        async def handler(scope, receive, send):
            with open(self.temp_path, "rb") as fh:
                fh.seek(55)
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"Content-Length", b"10")],
                    }
                )
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 200,
                        "count": 0,
                        "more_body": True,
                    }
                )
                state["position_after"] = fh.tell()
                await send({"type": "http.response.body", "body": b"0123456789"})
            done.set()

        _, head, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        self.assertEqual(body, b"0123456789")
        self.assertEqual(state["position_after"], 55)
        self.assertIn(b"Content-Length: 10", head)

    def test_count_defaults_to_end_of_file(self):
        done = threading.Event()
        start = 4000

        async def handler(scope, receive, send):
            with open(self.temp_path, "rb") as fh:
                fh.seek(start)
                await send({"type": "http.response.start", "status": 200})
                await send({"type": "http.response.zerocopysend", "file": fh})
            done.set()

        _, _, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        self.assertEqual(body, self.payload[start:])
        self.assertEqual(len(body), len(self.payload) - start)

    def test_repeated_interleaved_file_and_body_sends(self):
        done = threading.Event()

        async def handler(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            with open(self.temp_path, "rb") as fh:
                await send(
                    {"type": "http.response.body", "body": b"HEAD-", "more_body": True}
                )
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 10,
                        "count": 20,
                        "more_body": True,
                    }
                )
                await send(
                    {"type": "http.response.body", "body": b"-MID-", "more_body": True}
                )
                fh.seek(100)
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "count": 30,
                        "more_body": True,
                    }
                )
                await send({"type": "http.response.body", "body": b"-TAIL"})
            done.set()

        _, _, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        expected = (
            b"HEAD-" + self.payload[10:30] + b"-MID-" + self.payload[100:130] + b"-TAIL"
        )
        self.assertEqual(body, expected)

    def test_multiple_zerocopy_sends_same_response(self):
        done = threading.Event()

        async def handler(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            with open(self.temp_path, "rb") as fh:
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 0,
                        "count": 100,
                        "more_body": True,
                    }
                )
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 200,
                        "count": 50,
                    }
                )
            done.set()

        _, _, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        self.assertEqual(body, self.payload[0:100] + self.payload[200:250])

    def test_short_sends_and_eagain_preserve_exact_bytes(self):
        state = {}
        done = threading.Event()

        async def handler(scope, receive, send):
            with open(self.temp_path, "rb") as fh:
                await send({"type": "http.response.start", "status": 200})
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 0,
                        "count": len(self.payload),
                    }
                )
                state["final_position"] = fh.tell()
            done.set()

        fake = FakeSendfile(self.harness.server, short_cap=37, eagain_calls=6)
        _, _, body = self._run_request(self._handler(handler), fake=fake)
        self.assertTrue(done.wait(10))
        self.assertEqual(body, self.payload)
        self.assertGreater(len(fake.call_log), len(self.payload) // 37)
        # Explicit offset of 0 means the descriptor position never moves.
        self.assertEqual(state["final_position"], 0)
        # Every chunk the fake reported must be contiguous from offset 0.
        cursor = 0
        for offset, _count, sent in fake.call_log:
            self.assertEqual(offset, cursor)
            cursor += sent
        self.assertEqual(cursor, len(self.payload))

    def test_chunked_transfer_encoding_framing(self):
        # No Content-Length: Twisted selects chunked framing; the raw
        # sendfile bytes must still be framed correctly and dechunk by a
        # standard client.
        done = threading.Event()

        async def handler(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            with open(self.temp_path, "rb") as fh:
                await send(
                    {"type": "http.response.body", "body": b"pre:", "more_body": True}
                )
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 500,
                        "count": 256,
                        "more_body": True,
                    }
                )
                await send({"type": "http.response.body", "body": b":post"})
            done.set()

        self.harness.set_handler(self._handler(handler))
        fake = FakeSendfile(self.harness.server, short_cap=64)
        conn = HTTPConnection("127.0.0.1", self.harness.port, timeout=10)
        with mock.patch.object(zerocopy, "kernel_sendfile", fake):
            conn.request("GET", "/", headers={"Host": "example.com"})
            response = conn.getresponse()
            body = response.read()
        self.assertTrue(done.wait(10))
        self.assertEqual(response.getheader("Transfer-Encoding"), "chunked")
        self.assertEqual(body, b"pre:" + self.payload[500:756] + b":post")
        conn.close()

    ### HTTP rules

    def test_head_and_no_body_status_codes_send_no_bytes(self):
        for method, code in (("HEAD", 200), ("GET", 204), ("GET", 304)):
            with self.subTest(method=method, code=code):
                state = {}
                done = threading.Event()

                async def handler(scope, receive, send, code=code):
                    await send(
                        {
                            "type": "http.response.start",
                            "status": code,
                            "headers": [(b"Content-Length", b"4096")],
                        }
                    )
                    with open(self.temp_path, "rb") as fh:
                        fh.seek(10)
                        await send(
                            {
                                "type": "http.response.zerocopysend",
                                "file": fh,
                                "count": 100,
                            }
                        )
                        state["position"] = fh.tell()
                    done.set()

                self.harness.set_handler(self._handler(handler))
                sock = open_client(self.harness)
                try:
                    with mock.patch.object(
                        zerocopy, "kernel_sendfile", FakeSendfile(self.harness.server)
                    ):
                        sock.sendall(
                            (
                                "%s / HTTP/1.1\r\nHost: example.com\r\n"
                                "Connection: close\r\n\r\n" % method
                            ).encode("ascii")
                        )
                        head, body = read_raw_response(sock, expect_body=False)
                finally:
                    sock.close()
                self.assertTrue(done.wait(10))
                self.assertEqual(body, b"")
                self.assertIn(b"Content-Length: 4096", head)
                # No bytes transferred means the file position is untouched.
                self.assertEqual(state["position"], 10)

    def test_content_length_overrun_rejected(self):
        state = {}
        done = threading.Event()

        async def handler(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"Content-Length", b"10")],
                }
            )
            with open(self.temp_path, "rb") as fh:
                try:
                    await send(
                        {
                            "type": "http.response.zerocopysend",
                            "file": fh,
                            "offset": 0,
                            "count": 11,
                        }
                    )
                except ValueError as exc:
                    state["error"] = exc
            await send({"type": "http.response.body", "body": b"0123456789"})
            done.set()

        _, _, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        self.assertIsInstance(state["error"], ValueError)
        self.assertEqual(body, b"0123456789")

    ### Failure handling and resource cleanup

    def test_midtransfer_disconnect_cleans_up_and_keeps_file_open(self):
        state = {}
        done = threading.Event()

        async def handler(scope, receive, send):
            fh = open(self.temp_path, "rb")
            state["file"] = fh
            await send({"type": "http.response.start", "status": 200})
            try:
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 0,
                        "count": len(self.payload),
                    }
                )
            except Exception as exc:
                state["error"] = exc
            # The application still owns the descriptor: it must be open.
            state["still_open"] = os.fstat(fh.fileno()) is not None
            state["fileno"] = fh.fileno()
            fh.close()
            done.set()

        self.harness.set_handler(self._handler(handler))
        fake = FakeSendfile(self.harness.server, short_cap=128)
        sock = open_client(self.harness)
        client_port = sock.getsockname()[1]
        with mock.patch.object(zerocopy, "kernel_sendfile", fake):
            basic_request(sock)
            sock.settimeout(5)
            prefix = b""
            while b"\r\n\r\n" not in prefix:
                prefix += sock.recv(4096)
            # Read a little body then vanish mid-transfer.
            sock.recv(64)
            sock.close()
        self.assertTrue(done.wait(10))
        self.assertIsNotNone(state.get("error"))
        self.assertTrue(state["still_open"])
        # Give the reactor a moment to run connectionLost cleanup.
        deadline = time.time() + 5
        protocol = None
        while time.time() < deadline:
            protocol = self.harness.find_protocol(client_port)
            if protocol is None:
                break
            time.sleep(0.05)
        if protocol is not None:
            self._assert_no_dangling_producer(protocol)

    def test_cancelled_send_deregisters_producer_and_keeps_file_open(self):
        state = {}
        done = threading.Event()

        async def handler(scope, receive, send):
            fh = open(self.temp_path, "rb")
            await send({"type": "http.response.start", "status": 200})
            transfer = asyncio.ensure_future(
                send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 0,
                        "count": len(self.payload),
                    }
                )
            )
            # Let the producer register and enter its EAGAIN retry loop,
            # then cancel the in-flight ASGI send.
            await asyncio.sleep(0.05)
            transfer.cancel()
            try:
                await transfer
            except BaseException as exc:  # CancelledError on every runtime
                state["error"] = exc
            # The descriptor is application owned: it must remain open.
            state["still_open"] = os.fstat(fh.fileno()) is not None
            protocol = self._current_protocol(client_port)
            state["producer_clean"] = (
                self._producer_chain_is_clean(protocol)
                if protocol is not None
                else None
            )
            fh.close()
            done.set()

        self.harness.set_handler(self._handler(handler))
        # A long EAGAIN storm guarantees the transfer is still queued
        # when the cancellation lands.
        fake = FakeSendfile(self.harness.server, eagain_calls=10_000)
        sock = open_client(self.harness)
        client_port = sock.getsockname()[1]
        try:
            with mock.patch.object(zerocopy, "kernel_sendfile", fake):
                basic_request(sock)
                self.assertTrue(done.wait(10))
        finally:
            sock.close()
        self.assertIsInstance(state["error"], asyncio.CancelledError)
        self.assertTrue(state["still_open"])
        self.assertTrue(state["producer_clean"])
        self._assert_no_zerocopy_retry_timers()

    def _assert_no_dangling_producer(self, protocol):
        def _check():
            self.assertTrue(self._producer_chain_is_clean(protocol))

        run_in_reactor(_check)
        self._assert_no_zerocopy_retry_timers()

    def test_no_dangling_producer_after_success(self):
        done = threading.Event()
        state = {}

        async def handler(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            with open(self.temp_path, "rb") as fh:
                # Keep the response open so the request/channel are still
                # intact for inspection once the producer deregisters.
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 0,
                        "count": 128,
                        "more_body": True,
                    }
                )
            # The await above only returns after producer deregistration,
            # so assert the producer chain is clean while the request and
            # its protocol are still live in the reactor thread.
            protocol = self._current_protocol(client_port)
            state["clean"] = (
                self._producer_chain_is_clean(protocol)
                if protocol is not None
                else None
            )
            # Finish the response without adding further body bytes.
            await send({"type": "http.response.body", "body": b""})
            done.set()

        self.harness.set_handler(self._handler(handler))
        fake = FakeSendfile(self.harness.server, short_cap=16)
        sock = open_client(self.harness)
        client_port = sock.getsockname()[1]
        try:
            with mock.patch.object(zerocopy, "kernel_sendfile", fake):
                basic_request(sock)
                _, body = read_raw_response(sock)
        finally:
            sock.close()
        self.assertTrue(done.wait(10))
        self.assertEqual(body, self.payload[:128])
        self.assertTrue(state["clean"])
        self._assert_no_zerocopy_retry_timers()

    def _producer_chain_is_clean(self, protocol):
        if protocol.producer is not None:
            return False
        if getattr(protocol.channel, "_requestProducer", None) is not None:
            return False
        transport_producer = getattr(protocol.transport, "producer", None)
        return not isinstance(transport_producer, zerocopy.ZeroCopyFileProducer)

    def _current_protocol(self, client_port):
        # Deterministic lookup usable from inside a handler (which runs
        # in the reactor thread): match the server side peer port of
        # the test's own client socket.  Reaped protocols may linger in
        # server.connections with a recycled peer port, so require an
        # intact channel on a connected transport.
        for protocol in list(self.harness.server.connections.keys()):
            transport = getattr(protocol, "transport", None)
            # Reaped protocols have had their channel deleted; that also
            # disambiguates a recycled peer port from the live request.
            if transport is None or not hasattr(protocol, "channel"):
                continue
            try:
                if transport.getPeer().port == client_port:
                    return protocol
            except Exception:
                continue
        return None

    def _assert_no_zerocopy_retry_timers(self):
        delayed = run_in_reactor(reactor.getDelayedCalls)
        for call in delayed:
            self.assertNotIsInstance(
                getattr(getattr(call, "func", None), "__self__", None),
                zerocopy.ZeroCopyFileProducer,
            )

    def test_reactor_keeps_serving_other_connections_while_stalled(self):
        fast_done = threading.Event()
        slow_done = threading.Event()
        order = []

        async def slow_handler(scope, receive, send):
            order.append("slow-start")
            with open(self.temp_path, "rb") as fh:
                await send({"type": "http.response.start", "status": 200})
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 0,
                        "count": len(self.payload),
                    }
                )
            order.append("slow-done")
            slow_done.set()

        async def fast_handler(scope, receive, send):
            order.append("fast-start")
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"quick"})
            order.append("fast-done")
            fast_done.set()

        def route(scope, receive, send):
            if scope["path"] == "/slow":
                return slow_handler(scope, receive, send)
            return fast_handler(scope, receive, send)

        self.harness.set_handler(route)
        fake = FakeSendfile(self.harness.server, short_cap=64, eagain_calls=25)
        slow_sock = open_client(self.harness)
        # Drive the slow request inside a small worker so the main thread
        # can issue a second, independent request while it is stalled.
        slow_result = {}

        def slow_client():
            try:
                with mock.patch.object(zerocopy, "kernel_sendfile", fake):
                    basic_request(slow_sock, "/slow")
                    _, slow_body = read_raw_response(slow_sock)
                    slow_result["body"] = slow_body
            finally:
                slow_sock.close()

        worker = threading.Thread(target=slow_client)
        worker.start()
        # Wait until the slow transfer has actually entered EAGAIN retry.
        deadline = time.time() + 5
        while time.time() < deadline and fake.eagain_calls > 0:
            time.sleep(0.01)
        # Meanwhile a normal request on another connection must complete.
        fast_sock = open_client(self.harness)
        try:
            basic_request(fast_sock, "/fast")
            _, fast_body = read_raw_response(fast_sock)
        finally:
            fast_sock.close()
        self.assertTrue(fast_done.wait(10))
        self.assertEqual(fast_body, b"quick")
        worker.join(15)
        self.assertTrue(slow_done.wait(10))
        self.assertEqual(slow_result.get("body"), self.payload)
        self.assertLess(order.index("fast-done"), order.index("slow-done"))

    def test_send_before_response_start_is_rejected(self):
        state = {}
        done = threading.Event()

        async def handler(scope, receive, send):
            with open(self.temp_path, "rb") as fh:
                try:
                    await send(
                        {"type": "http.response.zerocopysend", "file": fh, "count": 1}
                    )
                except ValueError as exc:
                    state["error"] = exc
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})
            done.set()

        _, _, body = self._run_request(
            self._handler(handler), fake=FakeSendfile(self.harness.server)
        )
        self.assertTrue(done.wait(10))
        self.assertIsInstance(state["error"], ValueError)
        self.assertEqual(body, b"ok")

    @unittest.skipUnless(
        hasattr(os, "sendfile"), "real os.sendfile only exists on POSIX platforms"
    )
    def test_real_kernel_sendfile_end_to_end(self):
        # Runs on POSIX CI without any fake backend: proves the real
        # capability gate and syscall path work against a live socket.
        done = threading.Event()

        async def handler(scope, receive, send):
            self.assertIn(ZEROCOPY_EXTENSION, scope.get("extensions", {}))
            await send({"type": "http.response.start", "status": 200})
            with open(self.temp_path, "rb") as fh:
                await send(
                    {
                        "type": "http.response.zerocopysend",
                        "file": fh,
                        "offset": 12,
                        "count": 300,
                    }
                )
            done.set()

        _, _, body = self._run_request(self._handler(handler))
        self.assertTrue(done.wait(10))
        self.assertEqual(body, self.payload[12:312])


if __name__ == "__main__":
    unittest.main()
