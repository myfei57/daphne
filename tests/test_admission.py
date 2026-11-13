"""
Unit tests for the admission controller itself.

These tests never open a socket or start the reactor: they drive the
controller with ``asyncio.run`` so that queueing, timeouts, cancellation and
ticket accounting can be asserted deterministically.
"""

import asyncio
import unittest

from daphne.admission import (
    KIND_HTTP,
    KIND_WEBSOCKET,
    POLICY_REJECT,
    POLICY_WAIT,
    AdmissionConfig,
    AdmissionController,
    AdmissionRejected,
    Pool,
    RejectReason,
    Stage,
    TicketState,
)


def run(coro):
    return asyncio.run(coro)


class AdmissionConfigTests(unittest.TestCase):
    def test_disabled_by_default(self):
        config = AdmissionConfig()
        self.assertFalse(config.enabled)
        controller = AdmissionController(config)
        self.assertIsNone(controller.snapshot()["pools"]["http"]["limit"])
        self.assertEqual(controller.in_use(Pool.HTTP), 0)

    def test_enabled_when_any_capacity_set(self):
        self.assertTrue(AdmissionConfig(http_max_capacity=1).enabled)
        self.assertTrue(AdmissionConfig(websocket_pending_max_capacity=1).enabled)
        self.assertTrue(AdmissionConfig(websocket_active_max_capacity=1).enabled)

    def test_validation(self):
        with self.assertRaises(ValueError):
            AdmissionConfig(http_max_capacity=-1)
        with self.assertRaises(ValueError):
            AdmissionConfig(http_network_weight=0)
        with self.assertRaises(ValueError):
            AdmissionConfig(queue_timeout=0)
        with self.assertRaises(ValueError):
            AdmissionConfig(websocket_accept_full_policy="nope")
        with self.assertRaises(TypeError):
            AdmissionController(object())  # type: ignore[arg-type]


class HTTPAdmissionTests(unittest.TestCase):
    def test_hard_capacity_rejection(self):
        # Each HTTP ticket books network(1) + task(1) = 2 units.
        controller = AdmissionController(
            AdmissionConfig(http_max_capacity=2, max_queue_length=0)
        )

        async def scenario():
            ticket = await controller.admit(KIND_HTTP)
            self.assertEqual(controller.in_use(Pool.HTTP), 2)
            with self.assertRaises(AdmissionRejected) as caught:
                await controller.admit(KIND_HTTP)
            self.assertEqual(caught.exception.reason, RejectReason.CAPACITY)
            # A rejected request leaves no ticket behind.
            self.assertEqual(controller.stage_counters()[Stage.HTTP_TASK], 1)
            controller.release_all(ticket)
            self.assertEqual(controller.in_use(Pool.HTTP), 0)

        run(scenario())

    def test_network_and_task_shares_release_independently(self):
        controller = AdmissionController(AdmissionConfig(http_max_capacity=4))

        async def scenario():
            first = await controller.admit(KIND_HTTP)
            second = await controller.admit(KIND_HTTP)
            stages = controller.stage_counters()
            self.assertEqual(stages[Stage.HTTP_NETWORK], 2)
            self.assertEqual(stages[Stage.HTTP_TASK], 2)

            # Client disconnect: only the network share comes back. The task
            # share stays booked while the application coroutine is alive.
            controller.release_network(first)
            stages = controller.stage_counters()
            self.assertEqual(stages[Stage.HTTP_NETWORK], 1)
            self.assertEqual(stages[Stage.HTTP_TASK], 2)
            self.assertEqual(controller.in_use(Pool.HTTP), 3)

            # The application future ending releases the remaining task share.
            controller.release_task(first)
            stages = controller.stage_counters()
            self.assertEqual(stages[Stage.HTTP_NETWORK], 1)
            self.assertEqual(stages[Stage.HTTP_TASK], 1)

            # Idempotency: repeated calls never drive the counters negative.
            controller.release_network(first)
            controller.release_task(first)
            self.assertEqual(controller.in_use(Pool.HTTP), 2)

            controller.release_all(second)
            self.assertEqual(controller.in_use(Pool.HTTP), 0)
            self.assertEqual(controller._tickets, {})

        run(scenario())

    def test_task_release_without_prior_network_release(self):
        controller = AdmissionController(AdmissionConfig(http_max_capacity=2))

        async def scenario():
            ticket = await controller.admit(KIND_HTTP)
            # Application ends before the network close is processed.
            controller.release_task(ticket)
            self.assertEqual(controller.in_use(Pool.HTTP), 0)
            # Late network release must be a no-op.
            controller.release_network(ticket)
            self.assertEqual(controller.in_use(Pool.HTTP), 0)

        run(scenario())


class QueueTests(unittest.TestCase):
    def test_fifo_grant_order(self):
        controller = AdmissionController(
            AdmissionConfig(http_max_capacity=2, max_queue_length=10)
        )
        order = []

        async def scenario():
            holder = await controller.admit(KIND_HTTP)

            async def queued(name):
                ticket = await controller.admit(KIND_HTTP)
                order.append(name)
                return ticket

            first = asyncio.ensure_future(queued("first"))
            second = asyncio.ensure_future(queued("second"))
            await asyncio.sleep(0)
            self.assertEqual(controller.waiting_length(), 2)
            # Releasing one slot wakes the head waiter only.
            controller.release_all(holder)
            await asyncio.sleep(0)
            self.assertEqual(order, ["first"])
            first_ticket = first.result()
            controller.release_all(first_ticket)
            await second
            self.assertEqual(order, ["first", "second"])
            controller.release_all(second.result())
            self.assertEqual(controller.in_use(Pool.HTTP), 0)

        run(scenario())

    def test_queue_length_budget(self):
        controller = AdmissionController(
            AdmissionConfig(http_max_capacity=2, max_queue_length=1)
        )

        async def scenario():
            holder = await controller.admit(KIND_HTTP)
            waiter = asyncio.ensure_future(controller.admit(KIND_HTTP))
            await asyncio.sleep(0)
            self.assertEqual(controller.waiting_length(), 1)
            with self.assertRaises(AdmissionRejected) as caught:
                await controller.admit(KIND_HTTP)
            self.assertEqual(caught.exception.reason, RejectReason.QUEUE_LENGTH)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            controller.release_all(holder)
            self.assertEqual(controller.waiting_length(), 0)

        run(scenario())

    def test_queue_bytes_budget(self):
        config = AdmissionConfig(
            http_max_capacity=2,
            max_queue_bytes=2,
            http_network_weight=1,
            http_task_weight=1,
        )
        controller = AdmissionController(config)

        async def scenario():
            holder = await controller.admit(KIND_HTTP)  # weight 2
            waiter = asyncio.ensure_future(controller.admit(KIND_HTTP))
            await asyncio.sleep(0)
            self.assertEqual(controller.waiting_bytes(), 2)
            with self.assertRaises(AdmissionRejected) as caught:
                await controller.admit(KIND_HTTP)
            self.assertEqual(caught.exception.reason, RejectReason.QUEUE_BYTES)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            controller.release_all(holder)

        run(scenario())

    def test_queue_timeout(self):
        controller = AdmissionController(
            AdmissionConfig(http_max_capacity=2, max_queue_length=5, queue_timeout=0.02)
        )

        async def scenario():
            holder = await controller.admit(KIND_HTTP)
            with self.assertRaises(AdmissionRejected) as caught:
                await controller.admit(KIND_HTTP)
            self.assertEqual(caught.exception.reason, RejectReason.QUEUE_TIMEOUT)
            self.assertEqual(controller.wait_timeouts_total, 1)
            self.assertEqual(controller.waiting_length(), 0)
            controller.release_all(holder)

        run(scenario())

    def test_cancellation_removes_waiter(self):
        controller = AdmissionController(
            AdmissionConfig(http_max_capacity=2, max_queue_length=5)
        )

        async def scenario():
            holder = await controller.admit(KIND_HTTP)
            waiter = asyncio.ensure_future(controller.admit(KIND_HTTP))
            await asyncio.sleep(0)
            self.assertEqual(controller.waiting_length(), 1)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            # No leftover waiter and the queue byte budget is returned.
            self.assertEqual(controller.waiting_length(), 0)
            self.assertEqual(controller.waiting_bytes(), 0)
            controller.release_all(holder)

        run(scenario())

    def test_release_of_queued_ticket_cancels_wait(self):
        controller = AdmissionController(
            AdmissionConfig(http_max_capacity=2, max_queue_length=5)
        )

        async def scenario():
            holder = await controller.admit(KIND_HTTP)
            task = asyncio.ensure_future(controller.admit(KIND_HTTP))
            await asyncio.sleep(0)
            queued_ticket = next(iter(controller._waiters[Pool.HTTP])).ticket
            # Simulate a connection disappearing while still queued.
            controller.release_all(queued_ticket)
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(controller.waiting_length(), 0)
            controller.release_all(holder)

        run(scenario())


class WebSocketAdmissionTests(unittest.TestCase):
    def test_pending_capacity(self):
        controller = AdmissionController(
            AdmissionConfig(websocket_pending_max_capacity=1, max_queue_length=0)
        )

        async def scenario():
            ticket = await controller.admit(KIND_WEBSOCKET)
            self.assertEqual(controller.in_use(Pool.WS_PENDING), 1)
            with self.assertRaises(AdmissionRejected) as caught:
                await controller.admit(KIND_WEBSOCKET)
            self.assertEqual(caught.exception.reason, RejectReason.CAPACITY)
            controller.release(ticket)
            self.assertEqual(controller.in_use(Pool.WS_PENDING), 0)

        run(scenario())

    def test_accept_converts_pending_to_active(self):
        controller = AdmissionController(
            AdmissionConfig(
                websocket_pending_max_capacity=2,
                websocket_active_max_capacity=1,
            )
        )

        async def scenario():
            first = await controller.admit(KIND_WEBSOCKET)
            second = await controller.admit(KIND_WEBSOCKET)
            await controller.convert_to_active(first)
            self.assertEqual(controller.in_use(Pool.WS_PENDING), 1)
            self.assertEqual(controller.in_use(Pool.WS_ACTIVE), 1)
            self.assertEqual(first.state, TicketState.WS_ACTIVE)

            # Reject policy: full active pool fails the accept, the pending
            # ticket remains for the caller to release.
            with self.assertRaises(AdmissionRejected) as caught:
                await controller.convert_to_active(second)
            self.assertEqual(caught.exception.reason, RejectReason.WS_ACTIVE_FULL)
            self.assertEqual(second.state, TicketState.WS_PENDING)
            controller.release(second)
            controller.release(first)
            self.assertEqual(controller.in_use(Pool.WS_PENDING), 0)
            self.assertEqual(controller.in_use(Pool.WS_ACTIVE), 0)

        run(scenario())

    def test_accept_wait_policy_grants_on_release(self):
        controller = AdmissionController(
            AdmissionConfig(
                websocket_pending_max_capacity=2,
                websocket_active_max_capacity=1,
                websocket_accept_full_policy=POLICY_WAIT,
                max_queue_length=5,
            )
        )

        async def scenario():
            first = await controller.admit(KIND_WEBSOCKET)
            second = await controller.admit(KIND_WEBSOCKET)
            await controller.convert_to_active(first)

            async def convert_second():
                await controller.convert_to_active(second)
                return second.state

            task = asyncio.ensure_future(convert_second())
            await asyncio.sleep(0)
            self.assertEqual(controller.waiting_length(), 1)
            # Closing the active socket frees the slot; conversion completes
            # without ever double booking capacity.
            controller.release(first)
            state = await task
            self.assertEqual(state, TicketState.WS_ACTIVE)
            self.assertEqual(controller.in_use(Pool.WS_ACTIVE), 1)
            self.assertEqual(controller.in_use(Pool.WS_PENDING), 0)
            controller.release(second)
            self.assertEqual(controller.in_use(Pool.WS_ACTIVE), 0)

        run(scenario())

    def test_wait_policy_without_queue_budget_still_rejects(self):
        controller = AdmissionController(
            AdmissionConfig(
                websocket_pending_max_capacity=2,
                websocket_active_max_capacity=1,
                websocket_accept_full_policy=POLICY_WAIT,
                max_queue_length=0,
            )
        )

        async def scenario():
            first = await controller.admit(KIND_WEBSOCKET)
            second = await controller.admit(KIND_WEBSOCKET)
            await controller.convert_to_active(first)
            with self.assertRaises(AdmissionRejected) as caught:
                await controller.convert_to_active(second)
            self.assertEqual(
                caught.exception.reason, RejectReason.WS_ACTIVE_FULL
            )
            controller.release(second)
            controller.release(first)
            self.assertEqual(controller.in_use(Pool.WS_ACTIVE), 0)

        run(scenario())

    def test_duplicate_accept_is_idempotent(self):
        controller = AdmissionController(
            AdmissionConfig(
                websocket_pending_max_capacity=1,
                websocket_active_max_capacity=1,
            )
        )

        async def scenario():
            ticket = await controller.admit(KIND_WEBSOCKET)
            await controller.convert_to_active(ticket)
            await controller.convert_to_active(ticket)  # no-op
            self.assertEqual(controller.in_use(Pool.WS_ACTIVE), 1)
            controller.release(ticket)
            self.assertEqual(controller.in_use(Pool.WS_ACTIVE), 0)

        run(scenario())

    def test_pending_release_is_idempotent_across_terminal_paths(self):
        controller = AdmissionController(
            AdmissionConfig(websocket_pending_max_capacity=1)
        )

        async def scenario():
            ticket = await controller.admit(KIND_WEBSOCKET)
            # Timeout, client disconnect and reap may all fire.
            controller.release(ticket)
            controller.release(ticket)
            controller.release_all(ticket)
            self.assertEqual(controller.in_use(Pool.WS_PENDING), 0)
            self.assertEqual(controller._tickets, {})

        run(scenario())


class ShutdownAndMetricsTests(unittest.TestCase):
    def test_close_rejects_new_and_fails_waiters(self):
        controller = AdmissionController(
            AdmissionConfig(
                http_max_capacity=2,
                max_queue_length=5,
                websocket_accept_full_policy=POLICY_REJECT,
            )
        )

        async def scenario():
            holder = await controller.admit(KIND_HTTP)
            waiter = asyncio.ensure_future(controller.admit(KIND_HTTP))
            await asyncio.sleep(0)

            controller.close()
            self.assertTrue(controller.closed)
            with self.assertRaises(AdmissionRejected) as new_caught:
                await controller.admit(KIND_HTTP)
            self.assertEqual(new_caught.exception.reason, RejectReason.SHUTDOWN)
            with self.assertRaises(AdmissionRejected) as waiter_caught:
                await waiter
            self.assertEqual(waiter_caught.exception.reason, RejectReason.SHUTDOWN)
            self.assertEqual(controller.waiting_length(), 0)
            # Granted tickets survive close() and follow the normal flow.
            self.assertEqual(controller.in_use(Pool.HTTP), 2)
            controller.release_all(holder)
            # Idempotent close.
            controller.close()

        run(scenario())

    def test_metrics_use_only_finite_labels(self):
        controller = AdmissionController(
            AdmissionConfig(
                http_max_capacity=2,
                websocket_pending_max_capacity=2,
                websocket_active_max_capacity=2,
                max_queue_length=0,
            )
        )

        async def scenario():
            ticket = await controller.admit(KIND_HTTP)
            try:
                await controller.admit(KIND_WEBSOCKET)
            except AdmissionRejected:
                pass
            text = controller.render_metrics()
            # Every label value comes from a finite enum; no client data
            # (addresses, paths, headers) may appear anywhere in the output.
            self.assertIn('pool="http"', text)
            self.assertIn('stage="http_task"', text)
            self.assertIn('reason="capacity"', text)
            self.assertIn('dimension="length"', text)
            for forbidden in ("127.0.0.1", "client", "/path"):
                self.assertNotIn(forbidden, text)
            snapshot = controller.snapshot()
            self.assertEqual(
                set(snapshot["rejected_total"]),
                {reason.value for reason in RejectReason},
            )
            controller.release_all(ticket)

        run(scenario())

    def test_weights_are_server_fixed(self):
        # Even if a client were able to influence request data, the
        # controller only ever applies the configured integer weights.
        config = AdmissionConfig(
            http_max_capacity=6, http_network_weight=2, http_task_weight=1
        )
        controller = AdmissionController(config)

        async def scenario():
            ticket = await controller.admit(KIND_HTTP)
            # 3 units booked regardless of anything in the request.
            self.assertEqual(controller.in_use(Pool.HTTP), 3)
            controller.release_all(ticket)
            self.assertEqual(controller.in_use(Pool.HTTP), 0)

        run(scenario())


if __name__ == "__main__":
    unittest.main()
