"""
Optional per-request HTTP deadline policy.

This module implements :class:`HTTPDeadlinePolicy`, an opt-in addition to the
legacy coarse-grained ``http_timeout``.  A policy carries three *independent*
limits for a single HTTP request:

``first_response_timeout``
    Maximum time between the application being started for the request and
    the first ``http.response.start`` message.  Receiving ``response.start``
    ends this phase; the timer is not reinstalled for it afterwards.

``body_idle_timeout``
    Maximum idle gap between *observable application output* while the
    response body is being streamed.  Progress is governed by an explicit
    byte rule: only an ``http.response.body`` message whose ``body`` contains
    at least one byte (``len(body) > 0``) counts as progress.  Empty body
    messages, including ``more_body=True`` keep-alive style messages, never
    reset this timer, so an application cannot extend a request forever by
    emitting empty messages.

``total_timeout``
    Maximum wall-clock duration of the whole request, measured from
    application start.  It is never reset by any kind of progress.

Observability boundary
----------------------
``Request.write`` only hands data to Twisted's send buffer; entering that
buffer is not evidence that the bytes reached, or were drained by, the
client.  The idle deadline therefore observes the ASGI application-output
boundary (non-empty body messages accepted from the application) only.  No
producer/drain callbacks are registered here and no claim about client-side
transmission completion is made.

Timer management
----------------
Each request owns at most one active delayed call at any time.  Every
reschedule cancels the previous call (after bumping a generation counter) and
arms a single new call for the earliest applicable deadline.  Chunks can
therefore never accumulate uncancellable long-lived callbacks; the reactor's
own delayed-call heap remains the single bounded timer structure.  Every
callback captures its phase and generation, so stale callbacks left over from
a state transition, a finished request or a keep-alive reused connection can
never act on the new request.

All scheduling goes through an injectable clock whose production
implementation reads :func:`time.monotonic` (the reactor's own ``seconds()``
is wall-clock based and not suitable here); tests can substitute
``twisted.internet.task.Clock`` for deterministic virtual time.
"""

import logging
import math
import time

logger = logging.getLogger(__name__)


def _validate_timeout(name, value):
    """
    Validates a single optional timeout value.  ``None`` disables the limit;
    otherwise a finite positive number of seconds is required.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "%s must be a number of seconds or None, got %r" % (name, value)
        )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "%s must be a finite positive number of seconds, got %r" % (name, value)
        )
    return float(value)


class HTTPDeadlinePolicy:
    """
    Immutable description of the three optional per-request deadlines.

    Any individual limit may be ``None`` (disabled).  A policy with all three
    limits disabled behaves as if the feature was not configured.
    """

    def __init__(
        self,
        first_response_timeout=None,
        body_idle_timeout=None,
        total_timeout=None,
    ):
        self.first_response_timeout = _validate_timeout(
            "first_response_timeout", first_response_timeout
        )
        self.body_idle_timeout = _validate_timeout(
            "body_idle_timeout", body_idle_timeout
        )
        self.total_timeout = _validate_timeout("total_timeout", total_timeout)

    @property
    def enabled(self):
        """
        True if at least one of the three limits is configured.
        """
        return (
            self.first_response_timeout is not None
            or self.body_idle_timeout is not None
            or self.total_timeout is not None
        )

    @classmethod
    def create(cls, policy):
        """
        Normalises a constructor argument: ``None`` stays disabled, an
        existing policy is returned as-is.
        """
        if policy is None or isinstance(policy, cls):
            return policy
        raise TypeError(
            "http_deadline_policy must be an HTTPDeadlinePolicy instance or "
            "None, got %r" % (policy,)
        )

    def __repr__(self):
        return (
            "HTTPDeadlinePolicy(first_response_timeout=%r, body_idle_timeout=%r, "
            "total_timeout=%r)"
            % (
                self.first_response_timeout,
                self.body_idle_timeout,
                self.total_timeout,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, HTTPDeadlinePolicy):
            return NotImplemented
        return (
            self.first_response_timeout == other.first_response_timeout
            and self.body_idle_timeout == other.body_idle_timeout
            and self.total_timeout == other.total_timeout
        )

    def __hash__(self):
        return hash(
            (
                self.first_response_timeout,
                self.body_idle_timeout,
                self.total_timeout,
            )
        )


class MonotonicClock:
    """
    Clock adapter used in production.  Time reads come from the monotonic
    system clock while delayed calls are scheduled on the running reactor.
    """

    def __init__(self, reactor=None):
        if reactor is None:
            from twisted.internet import reactor

        self._reactor = reactor

    def seconds(self):
        return time.monotonic()

    def callLater(self, delay, func, *args, **kwargs):
        return self._reactor.callLater(delay, func, *args, **kwargs)


class RequestDeadline:
    """
    Per-request deadline state machine.

    Phases are ``PHASE_START`` (awaiting ``http.response.start``),
    ``PHASE_BODY`` (response started, awaiting body progress/finish) and
    ``PHASE_DONE`` (terminal; no timer is held).  ``generation`` is bumped on
    every reschedule so delayed calls from an earlier phase or generation
    become provably stale.
    """

    PHASE_START = "start"
    PHASE_BODY = "body"
    PHASE_DONE = "done"

    def __init__(self, request, policy, clock, manager=None):
        self.request = request
        self.policy = policy
        self._clock = clock
        self._manager = manager
        self.phase = self.PHASE_START
        self.generation = 0
        self.started_at = clock.seconds()
        self.last_progress_at = self.started_at
        self._call = None
        self._expired = False

    ### Lifecycle

    def arm(self):
        """
        Arms the initial set of deadlines in the start phase.
        """
        self._reschedule()

    def cancel(self):
        """
        Terminates all deadline handling for this request.  Idempotent: safe
        to call from ``finish``, ``connectionLost`` or server shutdown.
        """
        if self.phase == self.PHASE_DONE:
            return
        self.phase = self.PHASE_DONE
        self._clear_call()
        if self._manager is not None:
            self._manager.unregister(self)

    ### Progress events (real protocol progress only)

    def note_response_started(self):
        """
        Records ``http.response.start``.  This ends the first-response phase
        and moves the tracker into the body phase.  The first-response limit
        is never evaluated again.
        """
        if self.phase != self.PHASE_START:
            return
        self.phase = self.PHASE_BODY
        # The response start itself is progress for the body idle window.
        self.last_progress_at = self._clock.seconds()
        self._reschedule()

    def note_body(self, body):
        """
        Records an ``http.response.body`` message.

        Only non-empty bodies (``len(body) > 0``) count as progress.  Empty
        bodies, with or without ``more_body=True``, deliberately do not reset
        the idle timer so repeated empty messages can never extend a request.
        """
        if self.phase != self.PHASE_BODY:
            return
        if len(body) > 0:
            self.last_progress_at = self._clock.seconds()
            self._reschedule()

    ### Scheduling

    def _next_deadline(self):
        """
        Returns the earliest absolute deadline applicable to the current
        phase, or ``None`` if no limit applies.
        """
        candidates = []
        if self.phase == self.PHASE_START:
            if self.policy.first_response_timeout is not None:
                candidates.append(
                    self.started_at + self.policy.first_response_timeout
                )
        elif self.phase == self.PHASE_BODY:
            if self.policy.body_idle_timeout is not None:
                candidates.append(
                    self.last_progress_at + self.policy.body_idle_timeout
                )
        # The total deadline applies in every live phase and never resets.
        if self.policy.total_timeout is not None:
            candidates.append(self.started_at + self.policy.total_timeout)
        return min(candidates) if candidates else None

    def _clear_call(self):
        if self._call is not None:
            if self._call.active():
                self._call.cancel()
            self._call = None

    def _reschedule(self):
        if self.phase == self.PHASE_DONE:
            return
        deadline = self._next_deadline()
        if deadline is None:
            self._clear_call()
            return
        # Never schedule a negative delay; a deadline already in the past
        # fires on the next reactor iteration.
        delay = max(0.0, deadline - self._clock.seconds())
        self._clear_call()
        self.generation += 1
        generation = self.generation
        phase = self.phase
        self._call = self._clock.callLater(
            delay, self._on_expire, generation, phase
        )

    def _on_expire(self, generation, phase):
        """
        Delayed-call entry point.  Rejects any callback whose phase or
        generation does not match the current request state.
        """
        if (
            self.phase == self.PHASE_DONE
            or phase != self.phase
            or generation != self.generation
        ):
            # Stale callback from an earlier phase/generation (state
            # transition, finish, keep-alive reuse).  Never touch the request.
            return
        self._call = None
        self._expired = True
        # Move to the terminal phase first so re-entrant reply messages
        # triggered by the expiry action cannot reschedule anything.
        self.phase = self.PHASE_DONE
        if self._manager is not None:
            self._manager.unregister(self)
        try:
            if self.request._response_started:
                # Headers are already on the wire: a second status line would
                # be illegal, so the current transmission can only be cut.
                logger.warning(
                    "HTTP deadline exceeded while streaming response for %s; "
                    "terminating transport.",
                    getattr(self.request, "client_addr", None),
                )
                self.request._deadline_abort_transport()
            else:
                # Nothing has been sent yet: a legal 504 is still possible.
                logger.warning(
                    "HTTP deadline exceeded before response start for %s; "
                    "sending 504.",
                    getattr(self.request, "client_addr", None),
                )
                self.request._deadline_send_504()
        except Exception:
            logger.exception("HTTP deadline action failed for %r", self.request)

    @property
    def expired(self):
        return self._expired

    @property
    def has_active_call(self):
        return self._call is not None and self._call.active()


class DeadlineManager:
    """
    Server-wide registry of live :class:`RequestDeadline` trackers.

    The registry makes cleanup explicit and testable: every request must
    remove its tracker on finish, disconnect, expiry or shutdown, leaving the
    set empty once all requests have ended.
    """

    def __init__(self, clock=None):
        self.clock = clock if clock is not None else MonotonicClock()
        self._active = set()

    @property
    def active_count(self):
        return len(self._active)

    def active_trackers(self):
        return set(self._active)

    def create(self, request, policy):
        """
        Creates and arms a tracker for *request*.  Returns ``None`` (and
        allocates nothing) when no policy is configured, so unconfigured
        deployments keep their original behaviour.
        """
        policy = HTTPDeadlinePolicy.create(policy)
        if policy is None or not policy.enabled:
            return None
        tracker = RequestDeadline(request, policy, self.clock, manager=self)
        self._active.add(tracker)
        tracker.arm()
        return tracker

    def unregister(self, tracker):
        self._active.discard(tracker)

    def cancel_all(self):
        """
        Cancels every tracked request deadline, e.g. on server shutdown.
        """
        for tracker in list(self._active):
            tracker.cancel()
