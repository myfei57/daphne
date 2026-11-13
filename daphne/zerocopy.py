"""
Support for the ASGI ``http.response.zerocopysend`` extension.

The extension lets an ASGI application hand an open file object to the
server and have its bytes transferred with the operating system's
``sendfile(2)`` mechanism, without ever loading the file contents into
Python userspace buffers.

Capability policy (explicit, no fake zero copy)
-----------------------------------------------
The extension is only advertised on a connection when *all* of the
following hold:

* the platform actually exposes :func:`os.sendfile` (it is absent on
  Windows, for example);
* the request transport is a plain, unwrapped TCP socket transport
  (:class:`twisted.internet.tcp.Connection`).  TLS wrappers, HTTP/2
  stream channels, in-memory transports and anything else that does not
  map directly to a stream socket are excluded -- a ``sendfile`` to their
  descriptor would either bypass encryption or fail outright;
* the underlying socket has a valid, stream-oriented file descriptor.

TLS, HTTP/2 and platforms without a kernel ``sendfile`` therefore never
declare the extension, and no ordinary ``read``/``write`` fallback is
ever presented to the application as zero copy.  An application that
sends ``http.response.zerocopysend`` on a connection without the
extension gets a :class:`ValueError` instead of a silently downgraded
transfer.

File position rules follow the publicly documented behaviour of
:func:`os.sendfile`: with an explicit integer ``offset`` the file
descriptor's own position is left untouched (pread-style semantics);
when ``offset`` is omitted the kernel starts at, and updates, the file's
current position.
"""

import errno
import logging
import os
import socket
import stat
import sys

from twisted.internet import defer, error
from twisted.internet.interfaces import IPullProducer
from twisted.internet.tcp import Connection as TCPConnection
from twisted.python import failure
from zope.interface import implementer

logger = logging.getLogger(__name__)

ZEROCOPY_EXTENSION = "http.response.zerocopysend"

# The real kernel syscall.  ``None`` on platforms without os.sendfile
# (e.g. Windows); in that case the extension is never declared.
kernel_sendfile = getattr(os, "sendfile", None)

# Largest offset/count we accept.  Values must be non-negative and fit
# both Python's ssize_t and a 64 bit off_t; anything else is rejected
# before a single byte is written.
_MAX_OFFSET = min(sys.maxsize, (1 << 63) - 1)

# Maximum number of bytes moved by a single sendfile call.
_SEND_CHUNK = 1 << 20  # 1 MiB

# Maximum number of bytes moved during one producer turn, so the reactor
# keeps servicing other connections even on a fast local socket.
_RESUME_BUDGET = 4 * _SEND_CHUNK

# Adaptive backoff when the kernel reports EAGAIN/EWOULDBLOCK.
_INITIAL_RETRY_DELAY = 0.001
_MAX_RETRY_DELAY = 0.1

_SUPPORTED_SOCKET_FAMILIES = {
    socket.AF_INET,
    socket.AF_INET6,
    getattr(socket, "AF_UNIX", -1),
}


def zerocopy_capable(transport):
    """
    Return ``True`` only when ``transport`` is a real plaintext TCP
    socket transport on which a kernel ``sendfile`` can be issued.

    TLS wrapped transports, HTTP/2 channels and platforms without
    :func:`os.sendfile` deliberately return ``False`` -- the extension
    must not be advertised for them.
    """
    if kernel_sendfile is None:
        return False
    if not isinstance(transport, TCPConnection):
        # Excludes TLSMemoryBIOProtocol (TLS), H2Connection (HTTP/2),
        # test doubles, non-socket transports, ...
        return False
    try:
        sock = transport.getHandle()
    except Exception:
        return False
    if not isinstance(sock, socket.socket):
        return False
    try:
        fileno = sock.fileno()
    except OSError:
        return False
    if not isinstance(fileno, int) or fileno < 0:
        return False
    if sock.family not in _SUPPORTED_SOCKET_FAMILIES:
        return False
    if sock.type != socket.SOCK_STREAM:
        # sendfile(2) requires a stream oriented out descriptor.
        return False
    return True


def _validate_nonneg_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "http.response.zerocopysend '%s' must be an integer, got %r"
            % (name, value)
        )
    if value < 0:
        raise ValueError(
            "http.response.zerocopysend '%s' must not be negative, got %d"
            % (name, value)
        )
    if value > _MAX_OFFSET:
        raise OverflowError(
            "http.response.zerocopysend '%s' is too large: %d" % (name, value)
        )
    return value


def validate_zerocopy_message(message):
    """
    Validate a ``http.response.zerocopysend`` message before any byte is
    written.  Returns ``(file_object, in_fd, offset, count, stat_result)``
    where ``offset``/``count`` are either ``None`` or validated
    non-negative integers.

    Raises :class:`ValueError` (or :class:`OverflowError`) for non file
    objects, descriptors without ``fileno()``, negative or oversized
    offsets/counts and unusable descriptors.
    """
    file_obj = message.get("file")
    if file_obj is None or not hasattr(file_obj, "fileno") or not callable(
        file_obj.fileno
    ):
        raise ValueError(
            "http.response.zerocopysend 'file' must be an open file object "
            "with a fileno() method (e.g. not BytesIO)"
        )
    try:
        in_fd = file_obj.fileno()
    except Exception as exc:
        raise ValueError(
            "http.response.zerocopysend 'file' has no usable file "
            "descriptor: %s" % exc
        ) from exc
    if isinstance(in_fd, bool) or not isinstance(in_fd, int) or in_fd < 0:
        raise ValueError(
            "http.response.zerocopysend 'file' fileno() must return a "
            "non-negative integer descriptor, got %r" % (in_fd,)
        )

    offset = message.get("offset", None)
    if offset is not None:
        offset = _validate_nonneg_int(offset, "offset")

    count = message.get("count", None)
    if count is not None:
        count = _validate_nonneg_int(count, "count")

    # Prove the descriptor refers to an open file before we register a
    # producer or touch the response stream.
    try:
        stat_result = os.fstat(in_fd)
    except OSError as exc:
        raise ValueError(
            "http.response.zerocopysend 'file' descriptor %d is not usable: "
            "%s" % (in_fd, exc)
        ) from exc

    return file_obj, in_fd, offset, count, stat_result


def fragment_length(in_fd, offset, count, stat_result):
    """
    Determine how many bytes a zerocopysend message will put on the wire.

    Returns ``None`` when the length cannot be known up front (a
    ``count``-less transfer from a non regular file, e.g. a pipe).
    """
    if count is not None:
        return count
    if not stat.S_ISREG(stat_result.st_mode):
        return None
    if offset is not None:
        start = offset
    else:
        try:
            start = os.lseek(in_fd, 0, os.SEEK_CUR)
        except OSError:
            return None
    return max(0, stat_result.st_size - start)


@implementer(IPullProducer)
class ZeroCopyFileProducer:
    """
    Pull producer that streams a (range of a) file object to the client
    using ``sendfile(2)``.

    The producer is registered on the :class:`twisted.web.http.Request`
    as a non-streaming producer; Twisted wraps it in a cooperative task
    that drives :meth:`resumeProducing` one turn at a time, so the
    reactor never blocks reading the whole file and keeps processing
    other connections while a slow client drains the transfer.

    Short sends and ``EAGAIN``/``EWOULDBLOCK`` results advance (or
    retain) an explicit offset, so bytes are neither duplicated nor
    skipped.  All validation happens before this object is created; it
    never closes the application owned file.
    """

    def __init__(
        self,
        request,
        file_obj,
        in_fd,
        out_fd,
        offset,
        count,
        stat_result,
        backend,
        chunked,
    ):
        self._request = request
        # Keep a reference to the application owned file only for the
        # duration of the transfer; it is released on completion but
        # never closed by us.
        self._file = file_obj
        self._in_fd = in_fd
        self._out_fd = out_fd
        self._stat_result = stat_result
        self._backend = backend
        self._chunked = chunked

        # When an explicit offset was given we track it ourselves and
        # the descriptor's own position stays untouched (the syscall's
        # documented pread-style behaviour).  When offset is None the
        # backend gets None on every call and the kernel consumes and
        # advances the descriptor's current position.
        self._explicit_offset = offset is not None
        self._offset = offset
        self._remaining = count
        self._planned = fragment_length(in_fd, offset, count, stat_result)

        self._sent = 0
        self._done = False
        self._in_resume = False
        self._retry_call = None
        self._eagain_streak = 0

        self.completion = defer.Deferred(canceller=self._on_cancel)

    ### Producer driving

    def resumeProducing(self):
        if self._done or self._in_resume:
            return
        # An adaptive retry is armed after EAGAIN: wait for it instead of
        # busy-spinning against an unwritable socket on every reactor tick.
        if self._retry_call is not None:
            return
        request = self._request
        if request is None:
            return
        transport = getattr(request, "transport", None)
        if transport is None or getattr(transport, "disconnected", False):
            self._fail(
                failure.Failure(error.ConnectionLost("connection lost before sendfile"))
            )
            return
        # Never overtake bytes already buffered in the transport's
        # userspace output buffer (response/chunk headers written via
        # the normal write path must hit the wire first).
        if self._transport_has_pending(transport):
            return
        self._in_resume = True
        try:
            budget = _RESUME_BUDGET
            while not self._done and budget > 0:
                if self._remaining is not None:
                    if self._remaining == 0:
                        self._succeed()
                        return
                    chunk = min(_SEND_CHUNK, self._remaining)
                else:
                    chunk = _SEND_CHUNK
                offset_arg = self._offset if self._explicit_offset else None
                try:
                    sent = self._backend(
                        self._out_fd, self._in_fd, offset_arg, chunk
                    )
                except OSError as exc:
                    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        # Socket buffer full (slow client).  Keep the
                        # current offset untouched and wait for writability.
                        self._schedule_retry()
                        return
                    self._fail(failure.Failure(exc))
                    return
                except Exception as exc:
                    # Any other backend failure must surface on the
                    # completion Deferred instead of being swallowed by
                    # the cooperative producer driver and hanging forever.
                    self._fail(failure.Failure(exc))
                    return
                if sent == 0:
                    # EOF.
                    if self._remaining is not None and self._remaining > 0:
                        # The file shrank below the advertised range: the
                        # response stream cannot be made consistent.
                        self._fail(
                            failure.Failure(
                                error.ConnectionLost(
                                    "sendfile reached EOF with %d bytes still "
                                    "owed" % self._remaining
                                )
                            )
                        )
                        return
                    self._succeed()
                    return
                if sent < 0 or sent > chunk:
                    self._fail(
                        failure.Failure(
                            RuntimeError(
                                "sendfile backend returned invalid length %r"
                                % (sent,)
                            )
                        )
                    )
                    return
                self._sent += sent
                if self._explicit_offset:
                    self._offset += sent
                if self._remaining is not None:
                    self._remaining -= sent
                budget -= sent
                self._eagain_streak = 0
                if self._remaining is not None and self._remaining == 0:
                    self._succeed()
                    return
                # count is None: keep looping until EOF, bounded by the
                # per-turn budget so the reactor stays responsive.
        finally:
            self._in_resume = False

    def pauseProducing(self):
        # IPullProducer only requires resume/stop; backpressure is
        # applied by Twisted pausing the cooperative task.
        return None

    def stopProducing(self):
        # Called by Twisted when the connection is lost.  Stop driving
        # and release every server side resource, but do NOT close the
        # application owned file.
        if self._done:
            return
        reason = failure.Failure(
            error.ConnectionLost("connection lost during zerocopy sendfile")
        )
        self._cleanup(abort=False)
        self.completion.errback(reason)
        self._release()

    ### Internals

    def _on_cancel(self, deferred):
        # The ASGI application no longer awaits the transfer (its task
        # was cancelled).  Deregister the producer and stop scheduling
        # work; the application still owns (and must close) the file.
        # We only clean up here: Deferred.cancel() itself delivers the
        # cancellation failure to whoever still holds the Deferred.
        if self._done:
            return
        self._cleanup(abort=True)
        self._release()

    def _schedule_retry(self):
        # Import lazily so importing this module does not install a
        # default reactor before daphne.server picks its own.
        from twisted.internet import reactor

        self._cancel_retry()
        delay = min(
            _INITIAL_RETRY_DELAY * (2 ** self._eagain_streak), _MAX_RETRY_DELAY
        )
        self._eagain_streak += 1
        self._retry_call = reactor.callLater(delay, self._retry_fired)

    def _retry_fired(self):
        self._retry_call = None
        self.resumeProducing()

    def _cancel_retry(self):
        if self._retry_call is not None:
            call = self._retry_call
            self._retry_call = None
            if call.active():
                call.cancel()

    @staticmethod
    def _transport_has_pending(transport):
        data_buffer = getattr(transport, "dataBuffer", b"")
        offset = getattr(transport, "offset", 0)
        temp_len = getattr(transport, "_tempDataLen", 0)
        return (len(data_buffer) - offset) > 0 or temp_len > 0

    def _succeed(self):
        if self._planned is not None and self._sent != self._planned:
            self._fail(
                failure.Failure(
                    error.ConnectionLost(
                        "zerocopy transfer short by %d bytes"
                        % (self._planned - self._sent)
                    )
                )
            )
            return
        self._cleanup(abort=False)
        self.completion.callback(self._sent)
        self._release()

    def _fail(self, reason):
        self._cleanup(abort=True)
        self.completion.errback(reason)
        self._release()

    def _cleanup(self, abort):
        """
        Deregister the producer and release every server side resource.

        The application owned file is deliberately never closed.  When
        ``abort`` is true the connection is torn down so a partially
        written response cannot be reused for another request.
        """
        if self._done:
            return
        self._done = True
        self._cancel_retry()
        request = self._request
        if request is not None:
            # Deregister from the request/channel so no producer is left
            # hanging on this connection.
            try:
                request.unregisterProducer()
            except Exception:
                logger.debug("zerocopy producer unregister failed", exc_info=True)
            if abort:
                transport = getattr(request, "transport", None)
                if transport is not None:
                    try:
                        transport.abortConnection()
                    except Exception:
                        logger.debug(
                            "zerocopy abortConnection failed", exc_info=True
                        )
            elif self._chunked and self._planned:
                # Close the single chunk framing this fragment opened.
                try:
                    request.channel.write(b"\r\n")
                except Exception:
                    logger.debug("zerocopy chunk trailer failed", exc_info=True)
            try:
                request.sentLength += self._sent
            except Exception:
                pass

    def _release(self):
        # Break reference cycles; explicitly do not close self._file.
        self._request = None
        self._file = None

    @property
    def bytes_sent(self):
        return self._sent
