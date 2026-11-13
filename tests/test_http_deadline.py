"""
Tests for the optional per-request HTTPDeadlinePolicy.

Two layers are covered:

* deterministic unit tests driven by ``twisted.internet.task.Clock`` (virtual
  time) that assert exact deadline arithmetic, the explicit byte-based
  progress rule, generation/phase staleness and timer-entry cleanup;
* integration tests that boot real Daphne subprocesses with a custom ASGI
  application and verify observable protocol behaviour (504, transport
  termination, keep-alive independence, WebSocket isolation) over real
  sockets.
"""

import math
import socket
import time
import unittest
from http.client import HTTPConnection

from twisted.internet.task import Clock

from daphne.deadline import (
    DeadlineManager,
    HTTPDeadlinePolicy,
    MonotonicClock,
    RequestDeadline,
)
from daphne.http_protocol import WebRequest
from daphne.server import Server
from daphne.testing import (
    BaseDaphneTestingInstance,
    DaphneTestingInstance,
    EmptyChatterApplication,
    HealthyStreamApplication,
    SlowBodyApplication,
    SlowStartApplication,
)

from http_base import DaphneTestCase


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeRequest:
    """
    Minimal stand-in for WebRequest exposing only the surface touched by
    RequestDeadline expiry actions.
    """

    def __init__(self):
        self._response_started = False
        self.client_addr = ["127.0.0.1", 9999]
        self.status_504s = 0
        self.aborts = 0

    def _deadline_send_504(self):
        self.status_504s += 1
        self._response_started = True

    def _deadline_abort_transport(self):
        self.aborts += 1


class FakeQueue:
    def __init__(self):
        self.messages = []

    def put_nowait(self, message):
        self.messages.append(message)


class MockServer:
    def protocol_connected(self, *args, **kwargs):
        pass


class MockFactory:
    def __init__(self):
        self.server = MockServer()


class MockChannel:
    def __init__(self):
        self.factory = MockFactory()
        self.transport = None

    def getPeer(self, *args, **kwargs):
        return "peer"

    def getHost(self, *args, **kwargs):
        return "host"


class FakeReactor:
    def __init__(self):
        self.scheduled = []

    def callLater(self, delay, func, *args, **kwargs):
        self.scheduled.append((delay, func, args, kwargs))
        return self.scheduled[-1]


def make_tracker(policy):
    clock = Clock()
    manager = DeadlineManager(clock=clock)
    request = FakeRequest()
    tracker = manager.create(request, policy)
    return clock, manager, request, tracker


def start_response(tracker, request):
    """
    Mirrors WebRequest.handle_reply: the response-started flag is set on the
    request before the deadline tracker enters the body phase.
    """
    request._response_started = True
    tracker.note_response_started()


# ---------------------------------------------------------------------------
# Policy / clock unit tests
# ---------------------------------------------------------------------------


class HTTPDeadlinePolicyTests(unittest.TestCase):
    def test_defaults_are_disabled(self):
        policy = HTTPDeadlinePolicy()
        self.assertIsNone(policy.first_response_timeout)
        self.assertIsNone(policy.body_idle_timeout)
        self.assertIsNone(policy.total_timeout)
        self.assertFalse(policy.enabled)

    def test_validation_rejects_bad_values(self):
        for bad in (0, -1, float("nan"), float("inf"), float("-inf"), "5", True):
            with self.assertRaises((ValueError, TypeError)):
                HTTPDeadlinePolicy(first_response_timeout=bad)
            with self.assertRaises((ValueError, TypeError)):
                HTTPDeadlinePolicy(body_idle_timeout=bad)
            with self.assertRaises((ValueError, TypeError)):
                HTTPDeadlinePolicy(total_timeout=bad)

    def test_validation_accepts_positive_numbers(self):
        policy = HTTPDeadlinePolicy(
            first_response_timeout=1,
            body_idle_timeout=2.5,
            total_timeout=10,
        )
        self.assertEqual(policy.first_response_timeout, 1.0)
        self.assertEqual(policy.body_idle_timeout, 2.5)
        self.assertTrue(policy.enabled)
        self.assertEqual(
            policy,
            HTTPDeadlinePolicy(
                first_response_timeout=1.0,
                body_idle_timeout=2.5,
                total_timeout=10.0,
            ),
        )

    def test_create_normalises_input(self):
        self.assertIsNone(HTTPDeadlinePolicy.create(None))
        policy = HTTPDeadlinePolicy(total_timeout=3)
        self.assertIs(HTTPDeadlinePolicy.create(policy), policy)
        with self.assertRaises(TypeError):
            HTTPDeadlinePolicy.create({"total_timeout": 3})

    def test_monotonic_clock_uses_monotonic_time_and_reactor_calls(self):
        reactor = FakeReactor()
        clock = MonotonicClock(reactor)
        # Monotonic readings, never wall clock.
        self.assertTrue(math.isfinite(clock.seconds()))
        first = clock.seconds()
        time.sleep(0.001)
        self.assertGreaterEqual(clock.seconds(), first)
        marker = object()
        scheduled = clock.callLater(4, marker)
        self.assertEqual(scheduled[0], 4)
        self.assertIs(scheduled[1], marker)
        self.assertEqual(len(reactor.scheduled), 1)


# ---------------------------------------------------------------------------
# Virtual-clock deadline state machine tests
# ---------------------------------------------------------------------------


class RequestDeadlineClockTests(unittest.TestCase):
    def test_disabled_policy_allocates_nothing(self):
        clock = Clock()
        manager = DeadlineManager(clock=clock)
        self.assertIsNone(manager.create(FakeRequest(), None))
        self.assertIsNone(manager.create(FakeRequest(), HTTPDeadlinePolicy()))
        self.assertEqual(manager.active_count, 0)
        self.assertEqual(clock.getDelayedCalls(), [])

    def test_first_response_deadline_fires_at_exact_instant(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(first_response_timeout=5)
        )
        self.assertTrue(tracker.has_active_call)
        clock.advance(4.999)
        self.assertEqual(request.status_504s, 0)
        self.assertEqual(request.aborts, 0)
        # Expiry at the exact scheduled virtual instant.
        clock.advance(0.001)
        self.assertEqual(request.status_504s, 1)
        self.assertTrue(tracker.expired)
        self.assertEqual(tracker.phase, RequestDeadline.PHASE_DONE)
        # Terminal state: registry and clock both empty.
        self.assertEqual(manager.active_count, 0)
        self.assertEqual(clock.getDelayedCalls(), [])
        # Later ticks never trigger a second action.
        clock.advance(100)
        self.assertEqual(request.status_504s, 1)
        self.assertEqual(request.aborts, 0)

    def test_response_start_ends_first_phase_only(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(first_response_timeout=5, body_idle_timeout=10)
        )
        clock.advance(2)
        start_response(tracker, request)
        self.assertEqual(tracker.phase, RequestDeadline.PHASE_BODY)
        # Passing the old first-response deadline must do nothing now.
        clock.advance(4)
        self.assertEqual(request.status_504s, 0)
        # The body idle deadline (2 + 10 = 12) terminates the transmission.
        clock.advance(6)
        self.assertEqual(request.aborts, 1)
        self.assertEqual(request.status_504s, 0)
        self.assertEqual(manager.active_count, 0)

    def test_abort_after_start_never_writes_second_status_line(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(body_idle_timeout=1)
        )
        start_response(tracker, request)
        clock.advance(1)
        self.assertEqual(request.aborts, 1)
        self.assertEqual(request.status_504s, 0)

    def test_empty_bodies_never_reset_idle_even_repeatedly(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(body_idle_timeout=5)
        )
        start_response(tracker, request)
        # A stream of empty more_body=True messages right up to the window.
        for _ in range(4):
            clock.advance(1)
            tracker.note_body(b"")
        clock.advance(0.999)
        tracker.note_body(b"")
        self.assertEqual(request.aborts, 0)
        # Original idle instant still applies: empty chatter cannot extend.
        clock.advance(0.01)
        self.assertEqual(request.aborts, 1)
        self.assertEqual(manager.active_count, 0)

    def test_non_empty_body_resets_idle_per_byte_rule(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(body_idle_timeout=5)
        )
        start_response(tracker, request)
        clock.advance(4)
        tracker.note_body(b"")  # empty: no effect
        tracker.note_body(b"x")  # one real byte: progress, deadline moves to 9
        clock.advance(4.99)
        self.assertEqual(request.aborts, 0)
        clock.advance(0.02)
        self.assertEqual(request.aborts, 1)

    def test_total_timeout_caps_steady_progress(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(body_idle_timeout=10, total_timeout=3)
        )
        start_response(tracker, request)
        clock.advance(1)
        tracker.note_body(b"a")
        clock.advance(1)
        tracker.note_body(b"b")
        self.assertEqual(request.aborts, 0)
        # Total deadline at t=3 fires despite continuous byte progress.
        clock.advance(1)
        self.assertEqual(request.aborts, 1)

    def test_finish_at_deadline_instant_cancels_without_action(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(body_idle_timeout=5)
        )
        start_response(tracker, request)
        # Finish just before/at the scheduled expiry instant.
        clock.advance(4.99)
        tracker.cancel()
        clock.advance(0.02)
        self.assertEqual(request.aborts, 0)
        self.assertEqual(manager.active_count, 0)
        self.assertEqual(clock.getDelayedCalls(), [])

    def test_event_at_same_instant_makes_old_callback_stale(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(body_idle_timeout=5)
        )
        start_response(tracker, request)
        # Approach the original idle deadline; a non-empty chunk just before
        # that instant reschedules and bumps the generation.
        clock.advance(4.999)
        stale_generation = tracker.generation
        tracker.note_body(b"x")
        self.assertGreater(tracker.generation, stale_generation)
        # A late dispatch of the superseded callback must be a no-op.
        tracker._on_expire(stale_generation, RequestDeadline.PHASE_BODY)
        self.assertEqual(request.aborts, 0)
        # Crossing the original instant does nothing: the new window runs to 9.999.
        clock.advance(0.01)
        self.assertEqual(request.aborts, 0)
        clock.advance(5)
        self.assertEqual(request.aborts, 1)

    def test_empty_message_at_same_instant_does_not_save_request(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(body_idle_timeout=5)
        )
        start_response(tracker, request)
        # An empty message immediately before the instant does not move the
        # deadline, so expiry still happens at t=5.
        clock.advance(4.99)
        tracker.note_body(b"")
        clock.advance(0.02)
        self.assertEqual(request.aborts, 1)

    def test_phase_guard_rejects_callback_from_old_phase(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(first_response_timeout=5, body_idle_timeout=10)
        )
        start_generation = tracker.generation
        start_response(tracker, request)
        # Manually dispatch the superseded start-phase callback.
        tracker._on_expire(start_generation, RequestDeadline.PHASE_START)
        self.assertEqual(request.status_504s, 0)
        clock.advance(5)
        self.assertEqual(request.aborts, 0)
        self.assertEqual(tracker.phase, RequestDeadline.PHASE_BODY)

    def test_stale_callback_cannot_touch_keepalive_reused_connection(self):
        clock = Clock()
        manager = DeadlineManager(clock=clock)
        policy = HTTPDeadlinePolicy(body_idle_timeout=5)
        request_a = FakeRequest()
        tracker_a = manager.create(request_a, policy)
        start_response(tracker_a, request_a)
        generation_a = tracker_a.generation
        # Request A finishes; the connection is reused for request B.
        tracker_a.cancel()
        request_b = FakeRequest()
        tracker_b = manager.create(request_b, policy)
        # A stale callback from request A must never reach request B.
        tracker_a._on_expire(generation_a, RequestDeadline.PHASE_BODY)
        self.assertEqual(request_a.aborts, 0)
        self.assertEqual(request_b.aborts, 0)
        self.assertEqual(manager.active_trackers(), {tracker_b})
        tracker_b.cancel()
        self.assertEqual(manager.active_count, 0)
        self.assertEqual(clock.getDelayedCalls(), [])

    def test_only_one_active_call_no_matter_how_many_chunks(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(body_idle_timeout=10)
        )
        start_response(tracker, request)
        for index in range(25):
            tracker.note_body(b"chunk-%d" % index)
            self.assertTrue(tracker.has_active_call)
            # One tracker => at most one delayed call on the clock.
            self.assertEqual(len(clock.getDelayedCalls()), 1)
        tracker.cancel()
        self.assertEqual(clock.getDelayedCalls(), [])
        self.assertEqual(manager.active_count, 0)

    def test_cancel_all_keeps_clock_available_for_other_work(self):
        clock = Clock()
        manager = DeadlineManager(clock=clock)
        policy = HTTPDeadlinePolicy(total_timeout=5)
        manager.create(FakeRequest(), policy)
        manager.create(FakeRequest(), policy)
        self.assertEqual(manager.active_count, 2)
        # An unrelated timer, standing in for another connection's work.
        fired = []
        clock.callLater(7, fired.append, "still-served")
        manager.cancel_all()
        self.assertEqual(manager.active_count, 0)
        self.assertEqual(len(clock.getDelayedCalls()), 1)
        clock.advance(10)
        self.assertEqual(fired, ["still-served"])

    def test_events_after_expiry_are_ignored(self):
        clock, manager, request, tracker = make_tracker(
            HTTPDeadlinePolicy(first_response_timeout=1)
        )
        clock.advance(1)
        self.assertEqual(request.status_504s, 1)
        tracker.note_response_started()
        tracker.note_body(b"too late")
        tracker.note_body(b"")
        self.assertFalse(tracker.has_active_call)
        self.assertEqual(manager.active_count, 0)


class WebRequestDisconnectTests(unittest.TestCase):
    """
    http.disconnect must be delivered at most once per request.
    """

    def test_send_disconnect_is_idempotent(self):
        request = WebRequest(MockChannel())
        request.application_queue = FakeQueue()
        request.path = b"/"
        request.send_disconnect()
        request.send_disconnect()
        self.assertEqual(
            request.application_queue.messages, [{"type": "http.disconnect"}]
        )
        # A finished request loses its deadline tracker without error.
        self.assertIsNone(request._deadline)
        request._deadline_cancel()


class ServerDefaultsTests(unittest.TestCase):
    def test_unconfigured_server_keeps_legacy_defaults(self):
        server = Server(
            application=lambda scope, receive, send: None,
            endpoints=["tcp:10000"],
        )
        self.assertIsNone(server.http_timeout)
        self.assertIsNone(server.http_deadline_policy)

    def test_server_stores_policy_and_validates(self):
        policy = HTTPDeadlinePolicy(total_timeout=4)
        server = Server(
            application=lambda scope, receive, send: None,
            endpoints=["tcp:10000"],
            http_deadline_policy=policy,
        )
        self.assertEqual(server.http_deadline_policy, policy)
        with self.assertRaises(TypeError):
            Server(
                application=lambda scope, receive, send: None,
                endpoints=["tcp:10000"],
                http_deadline_policy=object(),
            )


# ---------------------------------------------------------------------------
# Real protocol integration tests
# ---------------------------------------------------------------------------


class DeadlineTestingInstance(BaseDaphneTestingInstance):
    """
    Testing instance booted with an explicit deadline policy and a custom
    module-level ASGI application.
    """

    def __init__(self, application, policy, http_timeout=None):
        super().__init__(
            application=application,
            http_deadline_policy=policy,
            http_timeout=http_timeout,
        )


class HTTPDeadlineIntegrationTests(DaphneTestCase):
    # Generous margins for CI scheduling jitter. The read timeout must exceed
    # the deadline so a response produced exactly on the deadline is observed.
    DEADLINE = 0.3
    SOCKET_TIMEOUT = 3.0
    READ_TIMEOUT = 1.0

    def _policy(self, **kwargs):
        kwargs.setdefault("first_response_timeout", None)
        kwargs.setdefault("body_idle_timeout", None)
        kwargs.setdefault("total_timeout", None)
        return HTTPDeadlinePolicy(**kwargs)

    def _raw_request(self, test_app):
        """
        Sends a plain HTTP/1.1 request over a raw socket and reads until the
        server closes the connection or the read budget is exhausted.
        Returns (data, server_closed, elapsed_seconds, first_byte_seconds).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.READ_TIMEOUT)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.connect((test_app.host, test_app.port))
        sock.sendall(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
        data = b""
        server_closed = False
        started = time.monotonic()
        first_byte_at = None
        while time.monotonic() - started < self.SOCKET_TIMEOUT:
            try:
                chunk = sock.recv(65535)
            except socket.timeout:
                break
            except ConnectionResetError:
                # A hard abortConnection() surfaces as RST (notably on
                # Windows): the server did terminate the transport.
                server_closed = True
                break
            if not chunk:
                server_closed = True
                break
            if first_byte_at is None:
                first_byte_at = time.monotonic() - started
            data += chunk
        elapsed = time.monotonic() - started
        sock.close()
        return data, server_closed, elapsed, first_byte_at

    def test_slow_start_gets_legal_504_and_reactor_kept_serving(self):
        policy = self._policy(first_response_timeout=self.DEADLINE)
        with DeadlineTestingInstance(SlowStartApplication(), policy) as test_app:
            data, closed, elapsed, first_byte = self._raw_request(test_app)
            # Twisted's standard reason phrase is "Gateway Time-out".
            self.assertIn(b"HTTP/1.1 504 Gateway Time-out", data)
            # The chunked response is fully written (final zero-length chunk),
            # i.e. a legal complete response rather than a torn connection.
            self.assertTrue(data.endswith(b"0\r\n\r\n"))
            self.assertIsNotNone(first_byte)
            self.assertGreaterEqual(first_byte, self.DEADLINE - 0.1)
            self.assertLess(elapsed, self.SOCKET_TIMEOUT)
            # A brand new connection is still accepted and promptly handled:
            # the reactor is alive and still enforcing the deadline.
            data2, _, _, first_byte2 = self._raw_request(test_app)
            self.assertIn(b"HTTP/1.1 504", data2)
            self.assertLess(first_byte2, self.SOCKET_TIMEOUT)

    def test_slow_body_aborts_transport_without_second_status_line(self):
        policy = self._policy(body_idle_timeout=self.DEADLINE)
        with DeadlineTestingInstance(SlowBodyApplication(), policy) as test_app:
            data, closed, elapsed, _ = self._raw_request(test_app)
            self.assertTrue(closed, "server was expected to abort the connection")
            self.assertEqual(data.count(b"HTTP/1.1"), 1)
            self.assertIn(b"200 OK", data)
            self.assertIn(b"partial", data)
            # No second status line may follow the started response.
            self.assertNotIn(b"504", data)
            self.assertGreaterEqual(elapsed, self.DEADLINE - 0.1)
            self.assertLess(elapsed, self.SOCKET_TIMEOUT)

    def test_repeated_empty_messages_cannot_extend_the_request(self):
        policy = self._policy(body_idle_timeout=self.DEADLINE)
        with DeadlineTestingInstance(
            EmptyChatterApplication(), policy
        ) as test_app:
            data, closed, elapsed, _ = self._raw_request(test_app)
            self.assertTrue(closed)
            self.assertIn(b"200 OK", data)
            # The empty-message chatter starts immediately, yet the original
            # idle deadline still terminates the request on time.
            self.assertLess(elapsed, self.DEADLINE + 1.0)

    def test_non_empty_byte_progress_keeps_stream_alive(self):
        policy = self._policy(body_idle_timeout=0.2, total_timeout=5)
        with DeadlineTestingInstance(
            HealthyStreamApplication(), policy
        ) as test_app:
            conn = HTTPConnection(test_app.host, test_app.port, timeout=3)
            conn.request("GET", "/")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            body = response.read()
            expected = b"".join(b"chunk%d" % i for i in range(5)) + b"done"
            self.assertEqual(body, expected)
            conn.close()

    def test_keepalive_reuse_runs_independent_trackers(self):
        policy = self._policy(body_idle_timeout=0.2, total_timeout=5)
        with DeadlineTestingInstance(
            HealthyStreamApplication(), policy
        ) as test_app:
            conn = HTTPConnection(test_app.host, test_app.port, timeout=3)
            for _ in range(2):
                conn.request("GET", "/")
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read().startswith(b"chunk0"))
            # The connection remains usable; old request timers never fire.
            conn.request("GET", "/")
            self.assertEqual(conn.getresponse().status, 200)
            conn.close()

    def test_pipelined_requests_both_complete_on_one_connection(self):
        policy = self._policy(body_idle_timeout=0.2, total_timeout=5)
        with DeadlineTestingInstance(
            HealthyStreamApplication(), policy
        ) as test_app:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((test_app.host, test_app.port))
            request = (
                b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
                b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
            )
            sock.sendall(request)
            data = b""
            while data.count(b"HTTP/1.1 200") < 2:
                chunk = sock.recv(65535)
                if not chunk:
                    break
                data += chunk
            self.assertEqual(data.count(b"HTTP/1.1 200"), 2)
            sock.close()

    def test_websocket_is_unaffected_by_http_deadlines(self):
        # Aggressive HTTP deadlines must never arm against an upgraded
        # connection; WebSocket timeouts remain independent.
        policy = self._policy(
            first_response_timeout=0.3, body_idle_timeout=0.3, total_timeout=3
        )
        with DaphneTestingInstance(http_deadline_policy=policy) as test_app:
            test_app.add_send_messages([{"type": "websocket.accept"}])
            sock, _ = self.websocket_handshake(test_app)
            # Stay quiet longer than every configured HTTP deadline.
            time.sleep(0.8)
            test_app.add_send_messages(
                [{"type": "websocket.send", "text": "cake"}]
            )
            self.websocket_send_frame(sock, "still alive?")
            self.assertEqual(self.websocket_receive_frame(sock), "cake")

    def test_policy_coexists_with_http_timeout_and_earlier_limit_wins(self):
        # Legacy http_timeout is far away; the tight deadline policy acts
        # first, and both mechanisms remain configured simultaneously.
        policy = self._policy(first_response_timeout=self.DEADLINE)
        with DeadlineTestingInstance(
            SlowStartApplication(), policy, http_timeout=100
        ) as test_app:
            data, _, _, first_byte = self._raw_request(test_app)
            self.assertIn(b"HTTP/1.1 504", data)
            self.assertLess(first_byte, 2.0)


if __name__ == "__main__":
    unittest.main()
