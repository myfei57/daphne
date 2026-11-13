"""
End-to-end admission tests using real HTTP and WebSocket connections.

A Daphne server runs in-process with the Twisted reactor on a background
thread and a controllable ASGI application whose per-connection gates let the
tests arrange queueing, accept races and disconnects deterministically. All
clients are plain sockets / http.client running in worker threads, so the
assertions exercise the full network stack rather than mocks.
"""

import asyncio
import base64
import http.client
import os
import socket
import threading
import time
import unittest
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait

# isort: off
# daphne.server must be imported before twisted.internet.reactor so its
# asyncio reactor is installed first; otherwise this module would bind to
# the default SelectReactor and talk to a reactor it does not run.
import daphne.server  # noqa: F401
from twisted.internet import reactor
# isort: on

from daphne.admission import AdmissionConfig, RejectReason, Stage
from daphne.server import Server

_REACTOR_THREAD = None


def setUpModule():
    """Twisted reactors cannot restart, so run one reactor for the module."""
    global _REACTOR_THREAD
    if not reactor.running:
        _REACTOR_THREAD = threading.Thread(
            target=reactor.run,
            kwargs={"installSignalHandlers": False},
            daemon=True,
        )
        _REACTOR_THREAD.start()
        deadline = time.time() + 5
        while time.time() < deadline and not reactor.running:
            time.sleep(0.02)
        if not reactor.running:
            raise RuntimeError("test reactor did not start")


def tearDownModule():
    if reactor.running:
        reactor.callFromThread(reactor.stop)
    if _REACTOR_THREAD is not None:
        _REACTOR_THREAD.join(timeout=10)


# --- Controllable ASGI application ----------------------------------------


class ControllableApplication:
    """
    Behavior is selected by the request path so one server can serve every
    scenario. Each connection gets its own asyncio gate so tests can hold
    HTTP requests and WebSocket accepts in any arrangement.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.instances = {}
        self.creation_order = []
        self._seq = 0

    async def __call__(self, scope, receive, send):
        with self.lock:
            self._seq += 1
            seq = self._seq
        record = {
            "seq": seq,
            "scope": scope,
            "gate": asyncio.Event(),
            "accepts": 0,
        }
        with self.lock:
            self.instances[seq] = record
            self.creation_order.append(seq)
        try:
            if scope["type"] == "http":
                await self._handle_http(record, scope, receive, send)
            else:
                await self._handle_websocket(record, scope, receive, send)
        finally:
            record["finished"] = True

    async def _handle_http(self, record, scope, receive, send):
        # Consume the http.request message(s).
        await receive()
        if scope["path"] == "/immediate":
            await self._send_http_ok(send)
            return
        if scope["path"] == "/raise":
            raise RuntimeError("controllable app failure")
        # /block: stay alive (ignoring client disconnects) until released.
        await record["gate"].wait()
        await self._send_http_ok(send)

    async def _handle_websocket(self, record, scope, receive, send):
        path = scope["path"]
        if path == "/raise":
            raise RuntimeError("controllable app failure")
        if path == "/reject":
            await send({"type": "websocket.close", "code": 1008})
            return
        if path == "/accept":
            await send({"type": "websocket.accept"})
        elif path in ("/manual", "/double"):
            await record["gate"].wait()
            await send({"type": "websocket.accept"})
            record["accepts"] += 1
            if path == "/double":
                # Duplicate accept must not book a second active ticket.
                await send({"type": "websocket.accept"})
                record["accepts"] += 1
        elif path == "/never":
            # Intentionally never accept; the handshake timeout must fire.
            await record["gate"].wait()
            await send({"type": "websocket.accept"})
        # Hold the accepted socket until the client goes away.
        while True:
            message = await receive()
            if message["type"] == "websocket.disconnect":
                break

    @staticmethod
    async def _send_http_ok(send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"Content-Type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"OK"})

    def release(self, seq):
        reactor.callFromThread(self.instances[seq]["gate"].set)

    def release_all(self):
        def _set_all():
            for record in self.instances.values():
                record["gate"].set()

        reactor.callFromThread(_set_all)

    def wait_instances(self, count, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if len(self.creation_order) >= count:
                    return list(self.creation_order)
            time.sleep(0.02)
        raise AssertionError(
            "expected %i application instances, got %i"
            % (count, len(self.creation_order))
        )


# --- In-process Daphne server ---------------------------------------------


class ServerInstance:
    """
    A Daphne server bound to an ephemeral port, sharing the module-level
    reactor. Start/stop are marshalled onto the reactor thread.
    """

    def __init__(self, app, config, **server_kwargs):
        kwargs = dict(
            signal_handlers=False,
            verbosity=0,
            websocket_connect_timeout=1,
            websocket_handshake_timeout=5,
            application_close_timeout=2,
            websocket_timeout=30,
            ping_interval=60,
            ping_timeout=60,
        )
        kwargs.update(server_kwargs)
        self.server = Server(
            application=app,
            endpoints=["tcp:0:interface=127.0.0.1"],
            admission_config=config,
            **kwargs,
        )
        self.host = "127.0.0.1"
        self.port = None

    def __enter__(self):
        reactor.callFromThread(self.server.start)
        deadline = time.time() + 10
        while time.time() < deadline:
            if self.server.listening_addresses:
                self.port = self.server.listening_addresses[0][1]
                return self
            time.sleep(0.02)
        reactor.callFromThread(self.server.stop_serving)
        raise RuntimeError("Daphne server did not start in time")

    def __exit__(self, *exc_info):
        reactor.callFromThread(self.server.stop_serving)

    @property
    def admission(self):
        return self.server.admission


# --- Client helpers --------------------------------------------------------


def http_request(port, path="/block", timeout=8, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path, headers=headers or {})
        response = conn.getresponse()
        body = response.read()
        return (
            response.status,
            {k.lower(): v for k, v in response.getheaders()},
            body,
        )
    finally:
        conn.close()


def raw_http(port, path="/block", timeout=5):
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.sendall(
        (
            "GET %s HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "Connection: close\r\n\r\n" % path
        ).encode("ascii")
    )
    return sock


def ws_connect(port, path="/manual", timeout=5, extra_headers=()):
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    lines = [
        "GET %s HTTP/1.1" % path,
        "Host: example.com",
        "Upgrade: websocket",
        "Connection: Upgrade",
        "Sec-WebSocket-Key: %s" % key,
        "Sec-WebSocket-Version: 13",
        "Origin: http://example.com",
    ]
    for name, value in extra_headers:
        lines.append("%s: %s" % (name, value))
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
    return sock


def read_response(sock, timeout=5):
    sock.settimeout(timeout)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    head = data.split(b"\r\n\r\n", 1)[0]
    status_line = head.split(b"\r\n", 1)[0].decode("ascii")
    status = int(status_line.split()[1])
    headers = {}
    for line in head.split(b"\r\n")[1:]:
        if b":" in line:
            name, value = line.split(b":", 1)
            headers[name.strip().lower().decode("ascii")] = value.strip().decode(
                "ascii"
            )
    return status, headers


def ws_close(sock):
    # Masked close frame with an empty payload.
    sock.sendall(b"\x88\x80\x00\x00\x00\x00")


def poll_until(predicate, timeout=8, interval=0.05, message="condition not met"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(message)


def snapshot(server):
    return server.admission.snapshot()


def assert_drained(server):
    def _drained():
        snap = snapshot(server)
        return (
            all(pool["in_use"] == 0 for pool in snap["pools"].values())
            and all(value == 0 for value in snap["stages"].values())
            and snap["queue"]["length"] == 0
            and server.admission.waiting_bytes() == 0
            and server.admission._tickets == {}
        )

    poll_until(_drained, timeout=10, message="admission capacity did not drain")


# --- Tests -----------------------------------------------------------------


class HTTPAdmissionIntegrationTests(unittest.TestCase):
    def test_http_capacity_emits_legal_503_and_caps_app_creation(self):
        app = ControllableApplication()
        # Two HTTP tickets at 2 units each fill the capacity.
        config = AdmissionConfig(http_max_capacity=4, max_queue_length=0)
        with ServerInstance(app, config) as server:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(http_request, server.port) for _ in range(3)]
                app.wait_instances(2)
                done, _ = futures_wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                self.assertEqual(len(done), 1)
                status, headers, _ = list(done)[0].result()
                self.assertEqual(status, 503)
                self.assertIn("retry-after", headers)

                # Only two application instances may ever be created.
                self.assertEqual(len(app.creation_order), 2)
                app.release_all()
                statuses = sorted(f.result(timeout=8)[0] for f in futures)
                self.assertEqual(statuses, [200, 200, 503])
            assert_drained(server)

    def test_http_queue_timeout_returns_503(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            http_max_capacity=2,
            max_queue_length=3,
            queue_timeout=0.5,
        )
        with ServerInstance(app, config) as server:
            with ThreadPoolExecutor(max_workers=2) as pool:
                holder = pool.submit(http_request, server.port)
                app.wait_instances(1)
                queued = pool.submit(http_request, server.port)
                poll_until(lambda: snapshot(server)["queue"]["length"] == 1)
                status, _, _ = queued.result(timeout=8)
                self.assertEqual(status, 503)
                snap = snapshot(server)
                self.assertGreaterEqual(
                    snap["rejected_total"][RejectReason.QUEUE_TIMEOUT.value], 1
                )
                self.assertGreaterEqual(snap["wait_timeouts_total"], 1)
                app.release_all()
                self.assertEqual(holder.result(timeout=8)[0], 200)
            assert_drained(server)

    def test_client_disconnect_while_queued_leaves_no_waiter(self):
        app = ControllableApplication()
        config = AdmissionConfig(http_max_capacity=2, max_queue_length=5)
        with ServerInstance(app, config) as server:
            with ThreadPoolExecutor(max_workers=1) as pool:
                holder = pool.submit(http_request, server.port)
                app.wait_instances(1)

                queued_socket = raw_http(server.port)
                poll_until(lambda: snapshot(server)["queue"]["length"] == 1)
                queued_socket.close()
                poll_until(lambda: snapshot(server)["queue"]["length"] == 0)

                # The freed queue slot is usable again.
                replacement = pool.submit(http_request, server.port, "/immediate")
                app.release_all()
                self.assertEqual(holder.result(timeout=8)[0], 200)
                self.assertEqual(replacement.result(timeout=8)[0], 200)
            assert_drained(server)

    def test_fifo_admission_order(self):
        app = ControllableApplication()
        config = AdmissionConfig(http_max_capacity=2, max_queue_length=10)
        with ServerInstance(app, config) as server:
            # Release each application gate as soon as its instance exists,
            # so the head waiter is always granted its slot in arrival order.
            def releaser():
                seen = set()
                deadline = time.time() + 10
                while time.time() < deadline and len(seen) < 5:
                    for seq in list(app.creation_order):
                        if seq not in seen:
                            seen.add(seq)
                            app.release(seq)
                    time.sleep(0.01)

            threading.Thread(target=releaser, daemon=True).start()

            def worker(index):
                status, _, _ = http_request(server.port)
                return index, status

            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = []
                for index in range(5):
                    futures.append(pool.submit(worker, index))
                    time.sleep(0.05)
                results = [future.result(timeout=10) for future in futures]

            # Admission (and therefore application creation) is FIFO.
            self.assertEqual(app.creation_order, [1, 2, 3, 4, 5])
            self.assertEqual([status for _, status in results], [200] * 5)
            assert_drained(server)

    def test_network_and_task_shares_are_separate(self):
        app = ControllableApplication()
        config = AdmissionConfig(http_max_capacity=10)
        with ServerInstance(app, config, application_close_timeout=1) as server:
            # A request whose network side we abandon while the app keeps running.
            orphan = raw_http(server.port)
            app.wait_instances(1)
            poll_until(lambda: snapshot(server)["stages"][Stage.HTTP_TASK.value] == 1)
            self.assertEqual(snapshot(server)["stages"][Stage.HTTP_NETWORK.value], 1)
            orphan.close()
            poll_until(
                lambda: snapshot(server)["stages"][Stage.HTTP_NETWORK.value] == 0
            )
            # Task share stays booked: not all budget is released early.
            self.assertEqual(snapshot(server)["stages"][Stage.HTTP_TASK.value], 1)
            self.assertEqual(snapshot(server)["pools"]["http"]["in_use"], 1)

            # A fresh request still fits because the network share was freed.
            self.assertEqual(http_request(server.port, "/immediate")[0], 200)

            # We never release this app's gate; the close timeout must kill
            # the task and return its share as well.
            poll_until(
                lambda: snapshot(server)["stages"][Stage.HTTP_TASK.value] == 0,
                timeout=8,
            )
            assert_drained(server)

    def test_client_self_reported_headers_cannot_change_weight(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            http_max_capacity=10,
            http_network_weight=2,
            http_task_weight=3,
        )
        with ServerInstance(app, config) as server:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    http_request,
                    server.port,
                    headers={"X-Admission-Weight": "999"},
                )
                app.wait_instances(1)
                poll_until(lambda: snapshot(server)["pools"]["http"]["in_use"] == 5)
                app.release_all()
                self.assertEqual(future.result(timeout=8)[0], 200)
            assert_drained(server)


class WebSocketAdmissionIntegrationTests(unittest.TestCase):
    def test_pending_flood_keeps_http_share_servable(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            http_max_capacity=4,
            websocket_pending_max_capacity=2,
            websocket_active_max_capacity=2,
            max_queue_length=0,
        )
        with ServerInstance(app, config, websocket_connect_timeout=15) as server:
            sockets = [ws_connect(server.port, "/manual") for _ in range(5)]
            app.wait_instances(2)
            time.sleep(0.3)  # let the rejections reach the clients

            outcomes = []
            for sock in sockets:
                try:
                    status, _ = read_response(sock, timeout=1.5)
                    outcomes.append(status)
                except socket.timeout:
                    # Still pending: the app has not accepted yet.
                    outcomes.append("PENDING")
            self.assertEqual(
                sorted(outcomes, key=lambda value: str(value)),
                [503, 503, 503, "PENDING", "PENDING"],
            )
            # A 101 must never have been written before rejection.
            self.assertEqual(app.creation_order, [1, 2])

            # The reserved HTTP share continues to serve short requests.
            self.assertEqual(http_request(server.port, "/immediate")[0], 200)
            self.assertEqual(http_request(server.port, "/immediate")[0], 200)

            # Let the pending handshakes complete normally.
            app.release_all()
            pending_sockets = [
                sock for sock, outcome in zip(sockets, outcomes) if outcome == "PENDING"
            ]
            for sock in pending_sockets:
                status, _ = read_response(sock, timeout=5)
                self.assertEqual(status, 101)
                ws_close(sock)
                sock.close()
            for sock in sockets:
                sock.close()
            assert_drained(server)

    def test_active_full_reject_fails_handshake_without_101(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            websocket_pending_max_capacity=2,
            websocket_active_max_capacity=1,
            max_queue_length=0,
        )
        with ServerInstance(app, config, websocket_connect_timeout=15) as server:
            first = ws_connect(server.port, "/manual")
            second = ws_connect(server.port, "/manual")
            app.wait_instances(2)
            self.assertEqual(snapshot(server)["stages"][Stage.WS_PENDING.value], 2)

            app.release(1)
            self.assertEqual(read_response(first)[0], 101)
            app.release(2)
            status, _ = read_response(second, timeout=5)
            self.assertEqual(status, 503)
            self.assertEqual(snapshot(server)["stages"][Stage.WS_ACTIVE.value], 1)
            self.assertEqual(snapshot(server)["stages"][Stage.WS_PENDING.value], 0)
            self.assertGreaterEqual(
                snapshot(server)["rejected_total"][RejectReason.WS_ACTIVE_FULL.value],
                1,
            )
            ws_close(first)
            first.close()
            second.close()
            assert_drained(server)

    def test_active_full_wait_policy_grants_after_close(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            websocket_pending_max_capacity=2,
            websocket_active_max_capacity=1,
            max_queue_length=5,
            queue_timeout=10,
            websocket_accept_full_policy="wait",
        )
        with ServerInstance(app, config, websocket_connect_timeout=15) as server:
            first = ws_connect(server.port, "/manual")
            second = ws_connect(server.port, "/manual")
            app.wait_instances(2)
            app.release_all()

            self.assertEqual(read_response(first)[0], 101)
            # The second accept is waiting for an active slot, so no 101 yet.
            with self.assertRaises(socket.timeout):
                second.settimeout(0.8)
                read_response(second, timeout=0.8)
            poll_until(
                lambda: snapshot(server)["pools"]["websocket_active"]["waiting"] == 1
            )

            ws_close(first)
            first.close()
            status, _ = read_response(second, timeout=5)
            self.assertEqual(status, 101)
            ws_close(second)
            second.close()
            assert_drained(server)

    def test_handshake_timeout_returns_pending_capacity(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            websocket_pending_max_capacity=2,
            websocket_active_max_capacity=2,
            max_queue_length=0,
        )
        with ServerInstance(
            app,
            config,
            websocket_connect_timeout=1,
            application_close_timeout=1,
        ) as server:
            sock = ws_connect(server.port, "/never")
            app.wait_instances(1)
            poll_until(lambda: snapshot(server)["stages"][Stage.WS_PENDING.value] == 1)
            status, _ = read_response(sock, timeout=8)
            # Handshake-level error, never a 101.
            self.assertNotEqual(status, 101)
            sock.close()
            assert_drained(server)

    def test_client_disconnect_before_accept_releases_once(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            websocket_pending_max_capacity=1,
            max_queue_length=0,
        )
        with ServerInstance(app, config, application_close_timeout=1) as server:
            sock = ws_connect(server.port, "/manual")
            app.wait_instances(1)
            poll_until(lambda: snapshot(server)["stages"][Stage.WS_PENDING.value] == 1)
            sock.close()
            poll_until(lambda: snapshot(server)["stages"][Stage.WS_PENDING.value] == 0)
            # Capacity must not go negative and must fully return to zero.
            self.assertGreaterEqual(
                snapshot(server)["pools"]["websocket_pending"]["in_use"], 0
            )
            assert_drained(server)

    def test_application_exception_releases_everything(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            http_max_capacity=4,
            websocket_pending_max_capacity=2,
            websocket_active_max_capacity=2,
        )
        with ServerInstance(app, config, application_close_timeout=1) as server:
            with ThreadPoolExecutor(max_workers=2) as pool:
                http_future = pool.submit(http_request, server.port, "/raise")
                ws_sock = ws_connect(server.port, "/raise")
                self.assertEqual(http_future.result(timeout=8)[0], 500)
                status, _ = read_response(ws_sock, timeout=5)
                self.assertEqual(status, 500)
                ws_sock.close()
            assert_drained(server)

    def test_duplicate_accept_books_single_active_ticket(self):
        app = ControllableApplication()
        config = AdmissionConfig(
            websocket_pending_max_capacity=2,
            websocket_active_max_capacity=2,
        )
        with ServerInstance(app, config) as server:
            sock = ws_connect(server.port, "/double")
            app.wait_instances(1)
            app.release(1)
            status, _ = read_response(sock, timeout=5)
            self.assertEqual(status, 101)
            # Only one active ticket despite two accept messages.
            self.assertEqual(snapshot(server)["stages"][Stage.WS_ACTIVE.value], 1)
            self.assertEqual(app.instances[1]["accepts"], 2)
            ws_close(sock)
            sock.close()
            assert_drained(server)


class ShutdownAndMetricsIntegrationTests(unittest.TestCase):
    def test_metrics_labels_stay_bounded_under_connection_flood(self):
        app = ControllableApplication()
        # Generous headroom: task shares are only reaped by the periodic
        # checker, so a fast flood must not trip its own capacity limit.
        config = AdmissionConfig(http_max_capacity=100, max_queue_length=0)
        with ServerInstance(app, config) as server:
            before = server.admission.render_metrics()
            for _ in range(20):
                status, _, _ = http_request(server.port, "/immediate", timeout=5)
                self.assertEqual(status, 200)
            assert_drained(server)
            after = server.admission.render_metrics()

            def label_keys(text):
                return sorted(
                    line.split("{", 1)[1].split("}", 1)[0]
                    for line in text.splitlines()
                    if "{" in line
                )

            self.assertEqual(label_keys(before), label_keys(after))
            # No client-identifying data may leak into metric labels.
            self.assertNotIn("127.0.0.1", after)
            for label in label_keys(after):
                self.assertIn(
                    label.split("=", 1)[0],
                    {"pool", "stage", "reason", "dimension"},
                )

    def test_shutdown_rejects_queue_and_cancels_waiters(self):
        app = ControllableApplication()
        config = AdmissionConfig(http_max_capacity=2, max_queue_length=5)
        instance = ServerInstance(app, config)
        instance.__enter__()
        try:
            server = instance.server
            # The production shutdown flow still drives the same callable
            # through the reactor's before-shutdown trigger.
            before_shutdown = [
                callable_obj
                for callable_obj, _args, _kwargs in reactor._eventTriggers[
                    "shutdown"
                ].before
            ]
            self.assertIn(server.kill_all_applications, before_shutdown)

            with ThreadPoolExecutor(max_workers=3) as pool:
                holder = pool.submit(http_request, instance.port)
                app.wait_instances(1)
                queued = [pool.submit(http_request, instance.port) for _ in range(2)]
                poll_until(lambda: snapshot(server)["queue"]["length"] == 2)

                # Run the exact shutdown handler the reactor would invoke.
                reactor.callFromThread(server.kill_all_applications)

                # Waiters were failed with a shutdown rejection; clients see
                # either the 503 response or a clean connection failure.
                for future in queued:
                    try:
                        status, _, _ = future.result(timeout=8)
                        self.assertEqual(status, 503)
                    except (http.client.HTTPException, OSError):
                        pass
                # No waiter may survive shutdown and new queueing is refused.
                self.assertEqual(server.admission.waiting_length(), 0)
                self.assertTrue(server.admission.closed)
                holder.cancel()
        finally:
            instance.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
