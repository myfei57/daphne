import logging
import time
import traceback
from urllib.parse import unquote

from autobahn.twisted.websocket import (
    ConnectionDeny,
    WebSocketServerFactory,
    WebSocketServerProtocol,
)
from twisted.internet import defer
from twisted.internet.defer import CancelledError

from .admission import AdmissionRejected, KIND_WEBSOCKET
from .utils import parse_x_forwarded_for

logger = logging.getLogger(__name__)


class WebSocketProtocol(WebSocketServerProtocol):
    """
    Protocol which supports WebSockets and forwards incoming messages to
    the websocket channels.
    """

    application_type = "websocket"

    # If we should send no more messages (e.g. we error-closed the socket)
    muted = False

    def onConnect(self, request):
        self.server = self.factory.server_class
        self.server.protocol_connected(self)
        self.request = request
        self.protocol_to_accept = None
        self.root_path = self.server.root_path
        self.socket_opened = time.time()
        self.last_ping = time.time()
        # Admission bookkeeping. The handshake Deferred must exist before the
        # admission coroutine can fail it, so it is created up front.
        self.handshake_deferred = defer.Deferred()
        self._admission_deferred = None
        self._ticket = None
        self._accept_started = False
        self._buffered_sends = []
        try:
            # Sanitize and decode headers, potentially extracting root path
            self.clean_headers = []
            for name, value in request.headers.items():
                name = name.encode("ascii")
                # Prevent CVE-2015-0219
                if b"_" in name:
                    continue
                if name.lower() == b"daphne-root-path":
                    self.root_path = unquote(value)
                else:
                    self.clean_headers.append((name.lower(), value.encode("latin1")))
            # Get client address if possible
            peer = self.transport.getPeer()
            host = self.transport.getHost()
            if hasattr(peer, "host") and hasattr(peer, "port"):
                self.client_addr = [str(peer.host), peer.port]
                self.server_addr = [str(host.host), host.port]
            else:
                self.client_addr = None
                self.server_addr = None

            if self.server.proxy_forwarded_address_header:
                self.client_addr, self.client_scheme = parse_x_forwarded_for(
                    dict(self.clean_headers),
                    self.server.proxy_forwarded_address_header,
                    self.server.proxy_forwarded_port_header,
                    self.server.proxy_forwarded_proto_header,
                    self.client_addr,
                )
            # Decode websocket subprotocol options
            self.subprotocols = []
            for header, value in self.clean_headers:
                if header == b"sec-websocket-protocol":
                    self.subprotocols = [
                        x.strip() for x in unquote(value.decode("ascii")).split(",")
                    ]
            # Make new application instance with scope
            self.path = request.path.encode("ascii")
            if self.server.admission is not None:
                # Admission runs before any application instance exists.
                # Rejection fails the still-open handshake with an HTTP error;
                # a 101 response is never sent first.
                self._admission_deferred = self.server.admission_deferred(
                    self.server.admission.admit(KIND_WEBSOCKET)
                )
                self._admission_deferred.addCallback(self._admissionSucceeded)
                self._admission_deferred.addErrback(self._admissionFailed)
            else:
                self._admissionSucceeded(None)
        except Exception:
            # Exceptions here are not displayed right, just 500.
            # Turn them into an ERROR log.
            logger.error(traceback.format_exc())
            raise

        # Make a deferred and return it - we'll either call it or err it later on
        return self.handshake_deferred

    def _build_scope(self):
        return {
            "type": "websocket",
            "path": unquote(self.path.decode("ascii")),
            "raw_path": self.path,
            "root_path": self.root_path,
            "headers": self.clean_headers,
            "query_string": self._raw_query_string,  # Passed by HTTP protocol
            "client": self.client_addr,
            "server": self.server_addr,
            "subprotocols": self.subprotocols,
        }

    def _admissionSucceeded(self, ticket):
        """
        Called when admission granted a (pending) WebSocket ticket.
        """
        self._ticket = ticket
        if ticket is not None:
            self.server.attach_admission_ticket(self, ticket)
        self.application_deferred = defer.maybeDeferred(
            self.server.create_application,
            self,
            self._build_scope(),
        )
        if self.application_deferred is not None:
            self.application_deferred.addCallback(self.applicationCreateWorked)
            self.application_deferred.addErrback(self.applicationCreateFailed)
        # Consume the admission Deferred's result.
        return None

    def _admissionFailed(self, failure):
        """
        Called when admission could not grant a pending ticket.
        """
        self._admission_deferred = None
        if failure.check(AdmissionRejected):
            logger.info(
                "WebSocket handshake rejected by admission controller: %s",
                failure.value.reason.value,
            )
            # Still pre-upgrade: fail the handshake, never emit 101.
            self._deny_handshake(503, "Service Unavailable")
            return None
        if failure.check(CancelledError):
            # Client disconnected while queued; the transport is already
            # going away. Just make sure the connection bookkeeping closes.
            logger.debug("WebSocket admission wait cancelled for %s", self)
            self.server.protocol_disconnected(self)
            return None
        logger.error(failure)
        return failure

    def applicationCreateWorked(self, application_queue):
        """
        Called when the background thread has successfully made the application
        instance.
        """
        # Store the application's queue
        self.application_queue = application_queue
        # Send over the connect message
        self.application_queue.put_nowait({"type": "websocket.connect"})
        self.server.log_action(
            "websocket",
            "connecting",
            {
                "path": self.request.path,
                "client": (
                    "%s:%s" % tuple(self.client_addr) if self.client_addr else None
                ),
            },
        )

    def applicationCreateFailed(self, failure):
        """
        Called when application creation fails.
        """
        logger.error(failure)
        self.server.release_admission_ticket(self)
        return failure

    ### Twisted event handling

    def onOpen(self):
        # Flush any messages the application sent before the accept completed
        # (serverSend auto-accepts while still connecting).
        for content, binary in self._buffered_sends:
            if binary:
                self.sendMessage(content, True)
            else:
                self.sendMessage(content.encode("utf8"), False)
        self._buffered_sends = []
        # Send news that this channel is open
        logger.debug("WebSocket %s open and established", self.client_addr)
        self.server.log_action(
            "websocket",
            "connected",
            {
                "path": self.request.path,
                "client": (
                    "%s:%s" % tuple(self.client_addr) if self.client_addr else None
                ),
            },
        )

    def onMessage(self, payload, isBinary):
        # If we're muted, do nothing.
        if self.muted:
            logger.debug("Muting incoming frame on %s", self.client_addr)
            return
        logger.debug("WebSocket incoming frame on %s", self.client_addr)
        self.last_ping = time.time()
        if isBinary:
            self.application_queue.put_nowait(
                {"type": "websocket.receive", "bytes": payload}
            )
        else:
            self.application_queue.put_nowait(
                {"type": "websocket.receive", "text": payload.decode("utf8")}
            )

    def onClose(self, wasClean, code, reason):
        """
        Called when Twisted closes the socket.
        """
        # Whatever ticket state we are in, release it exactly once. This
        # covers client disconnects before accept, handshake timeouts and
        # normal closes of accepted sockets.
        self.server.release_admission_ticket(self)
        self.server.protocol_disconnected(self)
        logger.debug("WebSocket closed for %s", self.client_addr)
        if not self.muted and hasattr(self, "application_queue"):
            self.application_queue.put_nowait(
                {"type": "websocket.disconnect", "code": code}
            )
        self.server.log_action(
            "websocket",
            "disconnected",
            {
                "path": self.request.path,
                "client": (
                    "%s:%s" % tuple(self.client_addr) if self.client_addr else None
                ),
            },
        )

    def connectionLost(self, reason):
        """
        Twisted transport hook. A client disconnect while the admission
        coroutine is still queued must cancel its bridge Deferred so no
        waiter is left behind.
        """
        admission_deferred = getattr(self, "_admission_deferred", None)
        if admission_deferred is not None:
            self._admission_deferred = None
            admission_deferred.cancel()
        super().connectionLost(reason)

    ### Internal event handling

    def handle_reply(self, message):
        if "type" not in message:
            raise ValueError("Message has no type defined")
        if message["type"] == "websocket.accept":
            self._begin_accept(message.get("subprotocol", None))
        elif message["type"] == "websocket.close":
            if self.state == self.STATE_CONNECTING:
                self.serverReject()
            else:
                self.serverClose(code=message.get("code", None))
        elif message["type"] == "websocket.send":
            if self.state == self.STATE_CONNECTING:
                raise ValueError("Socket has not been accepted, so cannot send over it")
            if message.get("bytes", None) and message.get("text", None):
                raise ValueError(
                    "Got invalid WebSocket reply message on %s - contains both bytes and text keys"
                    % (message,)
                )
            if message.get("bytes", None):
                self.serverSend(message["bytes"], True)
            if message.get("text", None):
                self.serverSend(message["text"], False)

    def handle_exception(self, exception):
        """
        Called by the server when our application tracebacks
        """
        # Free any capacity the connection is still holding.
        self.server.release_admission_ticket(self)
        if hasattr(self, "handshake_deferred") and not self.handshake_deferred.called:
            # If the handshake is still ongoing, we need to emit a HTTP error
            # code rather than a WebSocket one.
            self._deny_handshake(500, "Internal server error")
        else:
            self.sendCloseFrame(code=1011)

    def _deny_handshake(self, code, reason):
        """
        Fail the not-yet-upgraded handshake with an HTTP-level error. Never
        sends 101. Idempotent.
        """
        if not self.handshake_deferred.called:
            self.muted = True
            self.handshake_deferred.errback(ConnectionDeny(code=code, reason=reason))
        self.server.protocol_disconnected(self)
        logger.debug(
            "WebSocket %s handshake denied (%s) for %s",
            code,
            reason,
            self.client_addr,
        )

    def _begin_accept(self, subprotocol=None):
        """
        Entry point for websocket.accept. Duplicate accepts are ignored;
        the first one atomically converts the pending ticket into an
        active ticket before the 101 response is emitted.
        """
        if self._accept_started:
            logger.warning(
                "Duplicate websocket.accept ignored on %r", self
            )
            return
        self._accept_started = True
        if self._ticket is None or self.server.admission is None:
            self._finish_accept(subprotocol)
            return
        accept_deferred = self.server.admission_deferred(
            self.server.admission.convert_to_active(self._ticket)
        )
        accept_deferred.addCallback(lambda _: self._finish_accept(subprotocol))
        accept_deferred.addErrback(self._acceptFailed)

    def _finish_accept(self, subprotocol=None):
        # Only fire the handshake (101) after active capacity is secured.
        if not self.handshake_deferred.called:
            self.handshake_deferred.callback(subprotocol)
        logger.debug("WebSocket %s accepted by application", self.client_addr)

    def _acceptFailed(self, failure):
        """
        The pending -> active conversion failed. The handshake has not been
        upgraded, so respond with an HTTP error rather than a WebSocket frame.
        """
        if failure.check(AdmissionRejected):
            logger.info(
                "WebSocket accept rejected by admission controller: %s",
                failure.value.reason.value,
            )
            self._deny_handshake(503, "Service Unavailable")
            self.server.release_admission_ticket(self)
            return None
        if failure.check(CancelledError):
            # Shutdown or timeout raced with the accept wait; onClose or the
            # timeout path already handles bookkeeping.
            logger.debug("WebSocket accept wait cancelled for %s", self)
            return None
        logger.error(failure)
        return failure

    def serverReject(self):
        """
        Called when we get a message saying to reject the connection.
        """
        # Release the pending ticket first; idempotent against other paths.
        self.server.release_admission_ticket(self)
        self._deny_handshake(403, "Access denied")
        self.server.log_action(
            "websocket",
            "rejected",
            {
                "path": self.request.path,
                "client": (
                    "%s:%s" % tuple(self.client_addr) if self.client_addr else None
                ),
            },
        )

    def serverSend(self, content, binary=False):
        """
        Server-side channel message to send a message.
        """
        if self.state == self.STATE_CONNECTING:
            # The application sent data before an explicit accept: keep the
            # historical auto-accept behaviour but go through the admission
            # conversion, buffering the frame until the handshake completes.
            self._begin_accept(None)
            self._buffered_sends.append((content, binary))
            return
        logger.debug("Sent WebSocket packet to client for %s", self.client_addr)
        if binary:
            self.sendMessage(content, binary)
        else:
            self.sendMessage(content.encode("utf8"), binary)

    def serverClose(self, code=None):
        """
        Server-side channel message to close the socket
        """
        code = 1000 if code is None else code
        self.sendClose(code=code)

    ### Utils

    def duration(self):
        """
        Returns the time since the socket was opened
        """
        return time.time() - self.socket_opened

    def check_timeouts(self):
        """
        Called periodically to see if we should timeout something
        """
        # Web timeout checking
        if (
            self.duration() > self.server.websocket_timeout
            and self.server.websocket_timeout >= 0
        ):
            self.serverClose()
        # Ping check
        # If we're still connecting, deny the connection
        if self.state == self.STATE_CONNECTING:
            if self.duration() > self.server.websocket_connect_timeout:
                self.serverReject()
        elif self.state == self.STATE_OPEN:
            if (time.time() - self.last_ping) > self.server.ping_interval:
                self._sendAutoPing()
                self.last_ping = time.time()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    def __repr__(self):
        return f"<WebSocketProtocol client={self.client_addr!r} path={self.path!r}>"


class WebSocketFactory(WebSocketServerFactory):
    """
    Factory subclass that remembers what the "main"
    factory is, so WebSocket protocols can access it
    to get reply ID info.
    """

    protocol = WebSocketProtocol

    def __init__(self, server_class, *args, **kwargs):
        self.server_class = server_class
        WebSocketServerFactory.__init__(self, *args, **kwargs)

    def buildProtocol(self, addr):
        """
        Builds protocol instances. We use this to inject the factory object into the protocol.
        """
        try:
            protocol = super().buildProtocol(addr)
            protocol.factory = self
            return protocol
        except Exception:
            logger.error("Cannot build protocol: %s" % traceback.format_exc())
            raise
