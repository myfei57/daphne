"""
Application admission control.

Limits how many application instances Daphne creates before handing a
connection over to the ASGI application. Three independent capacities are
tracked so that long-lived or not-yet-accepted WebSocket connections cannot
starve short HTTP requests:

* ``http``               - HTTP requests, split into a *network* share
                           (released as soon as the client connection drops)
                           and a *task* share (released only once the
                           application coroutine ends).
* ``websocket_pending``  - WebSocket connections whose handshake has not
                           been accepted by the application yet.
* ``websocket_active``   - WebSocket connections after ``websocket.accept``.

The controller is loop agnostic: no asyncio primitives are created at
construction time and no blocking locks are used. All mutating code runs in
the single reactor/event-loop thread, so waiting is implemented with plain
futures rather than thread synchronisation.

Admission decisions only ever use the server-side configuration and the
connection kind (HTTP vs WebSocket, determined by Daphne itself). Request
headers or scope values supplied by the client can never influence a weight.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Optional

logger = logging.getLogger(__name__)


# --- Finite, bounded label vocabulary -------------------------------------


class Pool(str, Enum):
    """Capacity pools. Values are stable, finite metric labels."""

    HTTP = "http"
    WS_PENDING = "websocket_pending"
    WS_ACTIVE = "websocket_active"


class Stage(str, Enum):
    """Ticket stages. Values are stable, finite metric labels."""

    HTTP_NETWORK = "http_network"
    HTTP_TASK = "http_task"
    WS_PENDING = "websocket_pending"
    WS_ACTIVE = "websocket_active"


class RejectReason(str, Enum):
    """Finite set of rejection reasons (metric labels must stay bounded)."""

    CAPACITY = "capacity"
    QUEUE_LENGTH = "queue_length"
    QUEUE_BYTES = "queue_bytes"
    QUEUE_TIMEOUT = "queue_timeout"
    SHUTDOWN = "shutdown"
    WS_ACTIVE_FULL = "websocket_active_full"


POLICY_REJECT = "reject"
POLICY_WAIT = "wait"
_POLICIES = frozenset({POLICY_REJECT, POLICY_WAIT})

# Connection kinds, derived from the ASGI scope type Daphne builds itself.
KIND_HTTP = "http"
KIND_WEBSOCKET = "websocket"


class AdmissionRejected(Exception):
    """Raised when a connection cannot be admitted."""

    def __init__(self, reason: RejectReason, pool: Optional[Pool] = None):
        self.reason = reason
        self.pool = pool
        super().__init__("admission rejected: %s" % reason.value)


@dataclass
class AdmissionConfig:
    """
    Server-trusted admission configuration.

    Weights are fixed server-side integers (one connection == ``weight``
    capacity units). They are never derived from request data.
    """

    # Capacity limits. ``None`` disables the limit for that pool. The whole
    # controller is disabled unless at least one capacity is configured.
    http_max_capacity: Optional[int] = None
    websocket_pending_max_capacity: Optional[int] = None
    websocket_active_max_capacity: Optional[int] = None

    # Fixed, server-side weights per stage. Must be positive ints.
    http_network_weight: int = 1
    http_task_weight: int = 1
    websocket_pending_weight: int = 1
    websocket_active_weight: int = 1

    # Waiting queue budgets. ``None`` means unbounded for that dimension.
    max_queue_length: Optional[int] = None
    max_queue_bytes: Optional[int] = None
    queue_timeout: Optional[float] = None

    # What to do when ``websocket.accept`` arrives but the active pool is
    # full: reject the still-un-upgraded handshake or wait for a slot.
    websocket_accept_full_policy: str = POLICY_REJECT

    # Value suggested in the ``Retry-After`` header of HTTP 503 responses.
    retry_after: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("http_max_capacity", self.http_max_capacity),
            (
                "websocket_pending_max_capacity",
                self.websocket_pending_max_capacity,
            ),
            (
                "websocket_active_max_capacity",
                self.websocket_active_max_capacity,
            ),
            ("max_queue_length", self.max_queue_length),
            ("max_queue_bytes", self.max_queue_bytes),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError("%s must be a non-negative int or None" % name)
        for name, value in (
            ("http_network_weight", self.http_network_weight),
            ("http_task_weight", self.http_task_weight),
            ("websocket_pending_weight", self.websocket_pending_weight),
            ("websocket_active_weight", self.websocket_active_weight),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive int" % name)
        if self.queue_timeout is not None and (
            not isinstance(self.queue_timeout, (int, float)) or self.queue_timeout <= 0
        ):
            raise ValueError("queue_timeout must be a positive number or None")
        if self.websocket_accept_full_policy not in _POLICIES:
            raise ValueError(
                "websocket_accept_full_policy must be one of %s" % sorted(_POLICIES)
            )

    @property
    def enabled(self) -> bool:
        return (
            self.http_max_capacity is not None
            or self.websocket_pending_max_capacity is not None
            or self.websocket_active_max_capacity is not None
        )


class TicketState(str, Enum):
    NEW = "new"
    HTTP_HELD = "http_held"
    HTTP_NETWORK_RELEASED = "http_network_released"
    WS_PENDING = "websocket_pending"
    WS_ACTIVE = "websocket_active"
    RELEASED = "released"


@dataclass
class Ticket:
    """
    One admission grant.

    HTTP tickets carry two independently releasable shares (network and
    task). WebSocket tickets move from pending to active at most once. Every
    release is idempotent so the many terminal paths (handshake timeouts,
    client disconnects, application errors, duplicate accept messages,
    future reaping) can all release without double counting.
    """

    seq: int
    kind: str
    network_weight: int = 0
    task_weight: int = 0
    pending_weight: int = 0
    active_weight: int = 0
    state: TicketState = TicketState.NEW
    # Set while the ticket waits for an active slot during accept.
    active_waiting: bool = False

    @property
    def is_http(self) -> bool:
        return self.kind == KIND_HTTP


@dataclass
class _Waiter:
    seq: int
    pool: Pool
    weight: int
    future: "asyncio.Future[None]"
    ticket: Optional[Ticket]
    # True once the controller has granted the slot; guards against a late
    # timeout/cancellation touching an already consumed waiter.
    granted: bool = False


class AdmissionController:
    """
    Bounded admission gate with per-pool FIFO waiting queues.

    The public coroutines run on whatever asyncio loop is active; all
    synchronisation relies on executing inside a single event-loop thread.
    """

    def __init__(self, config: AdmissionConfig):
        if not isinstance(config, AdmissionConfig):
            raise TypeError("config must be an AdmissionConfig instance")
        self.config = config
        self._limits: Dict[Pool, Optional[int]] = {
            Pool.HTTP: config.http_max_capacity,
            Pool.WS_PENDING: config.websocket_pending_max_capacity,
            Pool.WS_ACTIVE: config.websocket_active_max_capacity,
        }
        self._in_use: Dict[Pool, int] = {
            Pool.HTTP: 0,
            Pool.WS_PENDING: 0,
            Pool.WS_ACTIVE: 0,
        }
        # Per-pool FIFO queues so a waiter for one pool cannot block a
        # different pool (cross-protocol fairness).
        self._waiters: Dict[Pool, Deque[_Waiter]] = {
            Pool.HTTP: Deque(),
            Pool.WS_PENDING: Deque(),
            Pool.WS_ACTIVE: Deque(),
        }
        self._tickets: Dict[int, Ticket] = {}
        self._queue_bytes = 0
        self._seq = 0
        self._closed = False

        # Bounded-label counters only.
        self.admitted_total: Dict[Pool, int] = {pool: 0 for pool in Pool}
        self.rejected_total: Dict[RejectReason, int] = {
            reason: 0 for reason in RejectReason
        }
        self.wait_timeouts_total = 0

    # --- Introspection ----------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def in_use(self, pool: Pool) -> int:
        return self._in_use[pool]

    def waiting_length(self) -> int:
        return sum(len(queue) for queue in self._waiters.values())

    def waiting_bytes(self) -> int:
        return self._queue_bytes

    def stage_counters(self) -> Dict[Stage, int]:
        """
        Per-stage in-use weights, including the split HTTP network/task
        shares. Only booked capacity is counted, so this always reconciles
        with the pool-level ``in_use`` numbers.
        """
        network = 0
        task = 0
        for ticket in self._tickets.values():
            if not ticket.is_http:
                continue
            if ticket.state in (
                TicketState.HTTP_HELD,
                TicketState.HTTP_NETWORK_RELEASED,
            ):
                task += ticket.task_weight
            if ticket.state == TicketState.HTTP_HELD:
                network += ticket.network_weight
        return {
            Stage.HTTP_NETWORK: network,
            Stage.HTTP_TASK: task,
            Stage.WS_PENDING: self._in_use[Pool.WS_PENDING],
            Stage.WS_ACTIVE: self._in_use[Pool.WS_ACTIVE],
        }

    def snapshot(self) -> dict:
        """Plain-dict view for tests and metrics."""
        stages = self.stage_counters()
        return {
            "closed": self._closed,
            "pools": {
                pool.value: {
                    "limit": self._limits[pool],
                    "in_use": self._in_use[pool],
                    "waiting": len(self._waiters[pool]),
                    "admitted_total": self.admitted_total[pool],
                }
                for pool in Pool
            },
            "stages": {stage.value: value for stage, value in stages.items()},
            "queue": {
                "length": self.waiting_length(),
                "bytes": self._queue_bytes,
            },
            "rejected_total": {
                reason.value: count for reason, count in self.rejected_total.items()
            },
            "wait_timeouts_total": self.wait_timeouts_total,
        }

    def render_metrics(self) -> str:
        """
        Prometheus text exposition using only finite labels (pool, stage,
        reason, dimension). Client addresses or other unbounded values are
        never used.
        """
        snap = self.snapshot()
        lines = [
            "# HELP daphne_admission_limit Configured capacity (-1 = unlimited).",
            "# TYPE daphne_admission_limit gauge",
        ]
        for pool_name, data in snap["pools"].items():
            limit = data["limit"] if data["limit"] is not None else -1
            lines.append('daphne_admission_limit{pool="%s"} %s' % (pool_name, limit))
        lines.append("# TYPE daphne_admission_in_use gauge")
        for pool_name, data in snap["pools"].items():
            lines.append(
                'daphne_admission_in_use{pool="%s"} %s' % (pool_name, data["in_use"])
            )
        lines.append("# TYPE daphne_admission_stage_in_use gauge")
        for stage_name, value in snap["stages"].items():
            lines.append(
                'daphne_admission_stage_in_use{stage="%s"} %s' % (stage_name, value)
            )
        lines.append("# TYPE daphne_admission_waiting gauge")
        for pool_name, data in snap["pools"].items():
            lines.append(
                'daphne_admission_waiting{pool="%s"} %s' % (pool_name, data["waiting"])
            )
        lines.append("# TYPE daphne_admission_admitted_total counter")
        for pool_name, data in snap["pools"].items():
            lines.append(
                'daphne_admission_admitted_total{pool="%s"} %s'
                % (pool_name, data["admitted_total"])
            )
        lines.append("# TYPE daphne_admission_rejected_total counter")
        for reason_name, count in snap["rejected_total"].items():
            lines.append(
                'daphne_admission_rejected_total{reason="%s"} %s' % (reason_name, count)
            )
        lines.extend(
            [
                "# TYPE daphne_admission_wait_timeouts_total counter",
                "daphne_admission_wait_timeouts_total %s" % snap["wait_timeouts_total"],
                "# TYPE daphne_admission_queue gauge",
                'daphne_admission_queue{dimension="length"} %s'
                % snap["queue"]["length"],
                'daphne_admission_queue{dimension="bytes"} %s' % snap["queue"]["bytes"],
            ]
        )
        return "\n".join(lines) + "\n"

    # --- Admission --------------------------------------------------------

    async def admit(self, kind: str) -> Ticket:
        """
        Admission entry point. Returns a held Ticket or raises
        AdmissionRejected.
        """
        if kind == KIND_HTTP:
            ticket = self._new_http_ticket()
            pool = Pool.HTTP
            weight = ticket.network_weight + ticket.task_weight
            held_state = TicketState.HTTP_HELD
        elif kind == KIND_WEBSOCKET:
            ticket = self._new_ws_ticket()
            pool = Pool.WS_PENDING
            weight = ticket.pending_weight
            held_state = TicketState.WS_PENDING
        else:
            raise ValueError("unknown admission kind %r" % kind)

        # Track the ticket from the start so safety-net releases can find
        # even queued (not yet granted) tickets.
        self._tickets[id(ticket)] = ticket

        try:
            if self._can_acquire(pool, weight):
                self._acquire(pool, weight)
                ticket.state = held_state
                self.admitted_total[pool] += 1
                return ticket

            # Hard rejection paths (no waiting possible).
            if self._closed:
                self._reject(RejectReason.SHUTDOWN, pool)
            if self.config.max_queue_length == 0:
                # Waiting is explicitly disabled: this is a hard cap.
                self._reject(RejectReason.CAPACITY, pool)
            if self.config.max_queue_length is not None and (
                self.waiting_length() >= self.config.max_queue_length
            ):
                self._reject(RejectReason.QUEUE_LENGTH, pool)
            if self.config.max_queue_bytes == 0:
                self._reject(RejectReason.CAPACITY, pool)
            if self.config.max_queue_bytes is not None and (
                self._queue_bytes + weight > self.config.max_queue_bytes
            ):
                self._reject(RejectReason.QUEUE_BYTES, pool)

            await self._enqueue_and_wait(pool, weight, ticket)
        except BaseException:
            # Rejection, timeout, cancellation or shutdown: while still in
            # NEW state the ticket never booked capacity, so drop it. A
            # granted ticket keeps its booking and must be released normally.
            if ticket.state == TicketState.NEW:
                self._forget(ticket)
            raise
        # The slot has been granted and booked by the pump.
        ticket.state = held_state
        self.admitted_total[pool] += 1
        return ticket

    async def convert_to_active(self, ticket: Ticket) -> None:
        """
        Atomically move a WebSocket ticket from the pending pool to the
        active pool. The 101 response must only be sent after this returns.
        """
        if ticket.kind != KIND_WEBSOCKET:
            raise ValueError("convert_to_active requires a WebSocket ticket")
        if ticket.state == TicketState.WS_ACTIVE:
            # Duplicate accept: idempotent no-op.
            return
        if ticket.state != TicketState.WS_PENDING:
            raise AdmissionRejected(RejectReason.WS_ACTIVE_FULL, Pool.WS_ACTIVE)

        weight = ticket.active_weight
        if self._can_acquire(Pool.WS_ACTIVE, weight):
            # Atomic swap: no await between the two bookings.
            self._release(Pool.WS_PENDING, ticket.pending_weight)
            self._acquire(Pool.WS_ACTIVE, weight)
            ticket.state = TicketState.WS_ACTIVE
            return

        if self.config.websocket_accept_full_policy == POLICY_REJECT:
            self._reject(RejectReason.WS_ACTIVE_FULL, Pool.WS_ACTIVE)
        # The wait policy still needs a queue budget to wait in.
        if self.config.max_queue_length == 0 or self.config.max_queue_bytes == 0:
            self._reject(RejectReason.WS_ACTIVE_FULL, Pool.WS_ACTIVE)

        ticket.active_waiting = True
        try:
            await self._enqueue_and_wait(Pool.WS_ACTIVE, weight, ticket)
        except BaseException:
            ticket.active_waiting = False
            raise
        ticket.active_waiting = False
        if ticket.state == TicketState.RELEASED:
            # The connection was released concurrently (timeout, client
            # disconnect, shutdown) right as the active slot was granted.
            # Undo that booking and refuse the upgrade.
            self._release(Pool.WS_ACTIVE, weight)
            self._reject(RejectReason.SHUTDOWN, Pool.WS_ACTIVE)
        # Grant only booked the active slot; release the pending share in
        # the same synchronous step so no stage is ever unaccounted.
        self._release(Pool.WS_PENDING, ticket.pending_weight)
        ticket.state = TicketState.WS_ACTIVE

    # --- Release (all idempotent) ----------------------------------------

    def release_network(self, ticket: Ticket) -> None:
        """Release the HTTP network share after a client disconnect/finish."""
        if not ticket.is_http or ticket.state != TicketState.HTTP_HELD:
            return
        self._release(Pool.HTTP, ticket.network_weight)
        ticket.state = TicketState.HTTP_NETWORK_RELEASED
        self._pump()

    def release_task(self, ticket: Ticket) -> None:
        """Release the HTTP task share once the application future has ended."""
        if not ticket.is_http:
            return
        if ticket.state == TicketState.HTTP_HELD:
            self._release(Pool.HTTP, ticket.network_weight + ticket.task_weight)
        elif ticket.state == TicketState.HTTP_NETWORK_RELEASED:
            self._release(Pool.HTTP, ticket.task_weight)
        else:
            return
        self._forget(ticket)
        self._pump()

    def release(self, ticket: Ticket) -> None:
        """Release whatever a WebSocket ticket currently holds."""
        if ticket.kind != KIND_WEBSOCKET:
            return
        if ticket.state == TicketState.NEW:
            # Queued but never granted (e.g. client gone during admission).
            self._cancel_waiter_for(ticket)
            self._forget(ticket)
            return
        if ticket.state == TicketState.WS_PENDING:
            if ticket.active_waiting:
                self._cancel_waiter_for(ticket)
            self._release(Pool.WS_PENDING, ticket.pending_weight)
        elif ticket.state == TicketState.WS_ACTIVE:
            self._release(Pool.WS_ACTIVE, ticket.active_weight)
        else:
            return
        self._forget(ticket)
        self._pump()

    def release_all(self, ticket: Ticket) -> None:
        """Safety net used when a connection is fully removed."""
        if ticket.state == TicketState.NEW:
            self._cancel_waiter_for(ticket)
            self._forget(ticket)
            return
        if ticket.kind == KIND_HTTP:
            self.release_network(ticket)
            self.release_task(ticket)
        else:
            self.release(ticket)

    # --- Shutdown ---------------------------------------------------------

    def close(self) -> None:
        """
        Stop admitting new connections and fail every queued waiter with a
        shutdown rejection. Already granted tickets are left untouched.
        """
        if self._closed:
            return
        self._closed = True
        for pool, queue in self._waiters.items():
            while queue:
                waiter = queue.popleft()
                self._queue_bytes -= waiter.weight
                if not waiter.future.done():
                    waiter.future.set_exception(
                        AdmissionRejected(RejectReason.SHUTDOWN, pool)
                    )

    # --- Internals --------------------------------------------------------

    def _new_http_ticket(self) -> Ticket:
        self._seq += 1
        return Ticket(
            seq=self._seq,
            kind=KIND_HTTP,
            network_weight=self.config.http_network_weight,
            task_weight=self.config.http_task_weight,
        )

    def _new_ws_ticket(self) -> Ticket:
        self._seq += 1
        return Ticket(
            seq=self._seq,
            kind=KIND_WEBSOCKET,
            pending_weight=self.config.websocket_pending_weight,
            active_weight=self.config.websocket_active_weight,
        )

    def _can_acquire(self, pool: Pool, weight: int) -> bool:
        # FIFO: never jump ahead of an existing waiter for the same pool.
        if self._waiters[pool]:
            return False
        limit = self._limits[pool]
        return limit is None or self._in_use[pool] + weight <= limit

    def _acquire(self, pool: Pool, weight: int) -> None:
        self._in_use[pool] += weight

    def _release(self, pool: Pool, weight: int) -> None:
        new_value = self._in_use[pool] - weight
        if new_value < 0:
            # Defensive: accounting bugs must never silently corrupt state.
            raise RuntimeError(
                "admission pool %s went negative (%r)" % (pool.value, new_value)
            )
        self._in_use[pool] = new_value

    def _reject(self, reason: RejectReason, pool: Pool) -> None:
        self.rejected_total[reason] += 1
        raise AdmissionRejected(reason, pool)

    async def _enqueue_and_wait(
        self, pool: Pool, weight: int, ticket: Optional[Ticket]
    ) -> None:
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[None]" = loop.create_future()
        self._seq += 1
        waiter = _Waiter(
            seq=self._seq,
            pool=pool,
            weight=weight,
            future=future,
            ticket=ticket,
        )
        self._waiters[pool].append(waiter)
        self._queue_bytes += weight
        try:
            if self.config.queue_timeout is None:
                await future
            else:
                try:
                    await asyncio.wait_for(future, self.config.queue_timeout)
                except asyncio.TimeoutError:
                    self.wait_timeouts_total += 1
                    self._reject(RejectReason.QUEUE_TIMEOUT, pool)
        finally:
            # Every exit path (grant, timeout, cancellation, shutdown)
            # removes exactly one waiter record: nothing is ever left behind.
            if not waiter.granted:
                try:
                    self._waiters[pool].remove(waiter)
                    self._queue_bytes -= weight
                except ValueError:
                    pass

    def _pump(self) -> None:
        """Grant FIFO head waiters while their pool has room."""
        for pool, queue in self._waiters.items():
            limit = self._limits[pool]
            while queue:
                waiter = queue[0]
                if limit is not None and self._in_use[pool] + waiter.weight > limit:
                    # Head of line: later waiters must not jump ahead.
                    break
                queue.popleft()
                self._queue_bytes -= waiter.weight
                waiter.granted = True
                self._acquire(pool, waiter.weight)
                if not waiter.future.done():
                    waiter.future.set_result(None)

    def _cancel_waiter_for(self, ticket: Ticket) -> None:
        for queue in self._waiters.values():
            for waiter in queue:
                if waiter.ticket is ticket and not waiter.future.done():
                    waiter.future.cancel()

    def _forget(self, ticket: Ticket) -> None:
        ticket.state = TicketState.RELEASED
        self._tickets.pop(id(ticket), None)
