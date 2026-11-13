# This has to be done first as Twisted is import-order-sensitive with reactors
import asyncio  # isort:skip
import os  # isort:skip
import sys  # isort:skip
import warnings  # isort:skip
from concurrent.futures import ThreadPoolExecutor  # isort:skip
from twisted.internet import asyncioreactor  # isort:skip

twisted_loop = asyncio.new_event_loop()
if "ASGI_THREADS" in os.environ:
    twisted_loop.set_default_executor(
        ThreadPoolExecutor(max_workers=int(os.environ["ASGI_THREADS"]))
    )

current_reactor = sys.modules.get("twisted.internet.reactor", None)
if current_reactor is not None:
    if not isinstance(current_reactor, asyncioreactor.AsyncioSelectorReactor):
        warnings.warn(
            "Something has already installed a non-asyncio Twisted reactor. Attempting to uninstall it; "
            + "you can fix this warning by importing daphne.server early in your codebase or "
            + "finding the package that imports Twisted and importing it later on.",
            UserWarning,
            stacklevel=2,
        )
        del sys.modules["twisted.internet.reactor"]
        asyncioreactor.install(twisted_loop)
else:
    asyncioreactor.install(twisted_loop)

import logging
import time
from concurrent.futures import CancelledError
from functools import partial

from twisted.internet import defer, reactor
from twisted.internet.endpoints import serverFromString
from twisted.logger import STDLibLogObserver, globalLogBeginner
from twisted.web import http

from .admission import AdmissionConfig, AdmissionController, KIND_HTTP
from .http_protocol import HTTPFactory
from .ws_protocol import WebSocketFactory

logger = logging.getLogger(__name__)


class Server:
    def __init__(
        self,
        application,
        endpoints=None,
        signal_handlers=True,
        action_logger=None,
        http_timeout=None,
        request_buffer_size=8192,
        websocket_timeout=86400,
        websocket_connect_timeout=20,
        ping_interval=20,
        ping_timeout=30,
        root_path="",
        proxy_forwarded_address_header=None,
        proxy_forwarded_port_header=None,
        proxy_forwarded_proto_header=None,
        verbosity=1,
        websocket_handshake_timeout=5,
        websocket_max_message_size=1024 * 1024,
        websocket_max_frame_size=1024 * 1024,
        application_close_timeout=10,
        ready_callable=None,
        server_name="daphne",
        admission_config=None,
    ):
        self.application = application
        self.endpoints = endpoints or []
        self.listeners = []
        self.listening_addresses = []
        # Optional application admission controller. Disabled (None) unless a
        # configuration with at least one capacity limit is supplied, so the
        # default server behaviour is unchanged.
        if admission_config is not None:
            if not isinstance(admission_config, AdmissionConfig):
                raise TypeError(
                    "admission_config must be an AdmissionConfig instance"
                )
            self.admission = (
                AdmissionController(admission_config)
                if admission_config.enabled
                else None
            )
        else:
            self.admission = None
        # protocol -> admission Ticket mapping
        self._admission_tickets = {}
        self.signal_handlers = signal_handlers
        self.action_logger = action_logger
        self.http_timeout = http_timeout
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.request_buffer_size = request_buffer_size
        self.proxy_forwarded_address_header = proxy_forwarded_address_header
        self.proxy_forwarded_port_header = proxy_forwarded_port_header
        self.proxy_forwarded_proto_header = proxy_forwarded_proto_header
        self.websocket_timeout = websocket_timeout
        self.websocket_connect_timeout = websocket_connect_timeout
        self.websocket_handshake_timeout = websocket_handshake_timeout
        self.websocket_max_message_size = websocket_max_message_size
        self.websocket_max_frame_size = websocket_max_frame_size
        self.application_close_timeout = application_close_timeout
        self.root_path = root_path
        self.verbosity = verbosity
        self.abort_start = False
        self.ready_callable = ready_callable
        self.server_name = server_name
        # Serving state. ``start`` flips this on; ``stop_serving`` flips it
        # off so the periodic checker chains terminate on their next tick.
        self._serving = False
        self._ports = []
        self._application_checker_call = None
        self._timeout_checker_call = None
        # Check our construction is actually sensible
        if not self.endpoints:
            logger.error("No endpoints. This server will not listen on anything.")
            sys.exit(1)

    def run(self):
        self.start()
        if not self.abort_start:
            # Trigger the ready flag if we had one
            if self.ready_callable:
                self.ready_callable()
            # Run the reactor
            reactor.run(installSignalHandlers=self.signal_handlers)

    def start(self):
        """
        Builds factories, schedules the maintenance loops and opens the
        endpoints, but does not run the reactor. Split out from ``run`` so
        several Server instances can share one reactor (e.g. in tests).
        """
        self._serving = True
        # A dict of protocol: {"application_instance":, "connected":, "disconnected":} dicts
        self.connections = {}
        self._admission_tickets = {}
        # Make the factory
        self.http_factory = HTTPFactory(self)
        self.ws_factory = WebSocketFactory(self, server=self.server_name)
        self.ws_factory.setProtocolOptions(
            autoPingTimeout=self.ping_timeout,
            allowNullOrigin=True,
            openHandshakeTimeout=self.websocket_handshake_timeout,
            maxMessagePayloadSize=self.websocket_max_message_size,
            maxFramePayloadSize=self.websocket_max_frame_size,
        )
        if self.verbosity <= 1:
            # Redirect the Twisted log to nowhere
            globalLogBeginner.beginLoggingTo(
                [lambda _: None], redirectStandardIO=False, discardBuffer=True
            )
        else:
            globalLogBeginner.beginLoggingTo([STDLibLogObserver(__name__)])

        # Detect what Twisted features are enabled
        if http.H2_ENABLED:
            logger.info("HTTP/2 support enabled")
        else:
            logger.info(
                "HTTP/2 support not enabled (install the http2 and tls Twisted extras)"
            )

        # Kick off the timeout loop
        self._application_checker_call = reactor.callLater(
            1, self.application_checker
        )
        self._timeout_checker_call = reactor.callLater(2, self.timeout_checker)

        for socket_description in self.endpoints:
            logger.info("Configuring endpoint %s", socket_description)
            ep = serverFromString(reactor, str(socket_description))
            listener = ep.listen(self.http_factory)
            listener.addCallback(self.listen_success)
            listener.addErrback(self.listen_error)
            self.listeners.append(listener)

        # Set the asyncio reactor's event loop as global
        # TODO: Should we instead pass the global one into the reactor?
        asyncio.set_event_loop(reactor._asyncioEventloop)

        # Verbosity 3 turns on asyncio debug to find those blocking yields
        if self.verbosity >= 3:
            asyncio.get_event_loop().set_debug(True)

        reactor.addSystemEventTrigger("before", "shutdown", self.kill_all_applications)

    def stop_serving(self):
        """
        Stops accepting new work and tears down this Server's resources
        without stopping the shared reactor. Used by the in-process tests;
        the production shutdown path remains ``kill_all_applications`` via
        the reactor's before-shutdown trigger.
        """
        if not self._serving:
            return
        self._serving = False
        # Admission: reject new queueing and wake every waiting connection.
        if self.admission is not None:
            self.admission.close()
        for port in self._ports:
            port.stopListening()
        self._ports = []
        # Cancel live application work; the final checker ticks release the
        # remaining admission tickets and remove the connection records.
        for details in list(self.connections.values()):
            application_instance = details.get("application_instance")
            if (
                application_instance is not None
                and not application_instance.done()
            ):
                application_instance.cancel()

    def listen_success(self, port):
        """
        Called when a listen succeeds so we can store port details (if there are any)
        """
        self._ports.append(port)
        if hasattr(port, "getHost"):
            host = port.getHost()
            if hasattr(host, "host") and hasattr(host, "port"):
                self.listening_addresses.append((host.host, host.port))
                logger.info(
                    "Listening on TCP address %s:%s",
                    port.getHost().host,
                    port.getHost().port,
                )

    def listen_error(self, failure):
        logger.critical("Listen failure: %s", failure.getErrorMessage())
        self.stop()

    def stop(self):
        """
        Force-stops the server.
        """
        if reactor.running:
            reactor.stop()
        else:
            self.abort_start = True

    ### Protocol handling

    def protocol_connected(self, protocol):
        """
        Adds a protocol as a current connection.
        """
        if protocol in self.connections:
            raise RuntimeError("Protocol %r was added to main list twice!" % protocol)
        self.connections[protocol] = {"connected": time.time()}

    def protocol_disconnected(self, protocol):
        # Set its disconnected time (the loops will come and clean it up)
        # Do not set it if it is already set. Overwriting it might
        # cause it to never be cleaned up.
        # See https://github.com/django/channels/issues/1181
        if "disconnected" not in self.connections[protocol]:
            self.connections[protocol]["disconnected"] = time.time()
        # Admission: the network side is gone. For HTTP only the network
        # share is released here; the task share stays booked until the
        # application future actually ends. WebSocket tickets are released
        # wholesale (pending or active), idempotently.
        ticket = self._admission_tickets.get(protocol)
        if ticket is not None and self.admission is not None:
            if ticket.kind == KIND_HTTP:
                self.admission.release_network(ticket)
            else:
                self.admission.release(ticket)

    ### Admission ticket handling

    def attach_admission_ticket(self, protocol, ticket):
        """
        Records the admission ticket granted for a protocol so the various
        terminal paths can release it exactly once.
        """
        self._admission_tickets[protocol] = ticket

    def get_admission_ticket(self, protocol):
        return self._admission_tickets.get(protocol)

    def release_admission_ticket(self, protocol):
        """
        Safety-net release used when a protocol object is fully removed.
        Staged releases earlier on are idempotent, so this only frees
        shares that are still held.
        """
        if self.admission is None:
            return
        ticket = self._admission_tickets.pop(protocol, None)
        if ticket is not None:
            self.admission.release_all(ticket)

    def admission_metrics(self):
        """
        Returns the admission controller's metric snapshot, or None when
        admission control is disabled. Labels are bounded by design.
        """
        if self.admission is None:
            return None
        return self.admission.snapshot()

    def admission_deferred(self, coroutine):
        """
        Bridges an admission controller coroutine (which awaits asyncio
        primitives) to a Twisted Deferred running natively on the reactor's
        asyncio event loop. Cancelling the Deferred cancels the underlying
        task, so queued waiters never outlive their connection.
        """
        loop = asyncio.get_event_loop()
        task = loop.create_task(coroutine)
        return defer.Deferred.fromFuture(task)

    ### Internal event/message handling

    def create_application(self, protocol, scope):
        """
        Creates a new application instance that fronts a Protocol instance
        for one of our supported protocols. Pass it the protocol,
        and it will work out the type, supply appropriate callables, and
        return you the application's input queue
        """
        # Make sure the protocol has not had another application made for it
        assert "application_instance" not in self.connections[protocol]
        # Make an instance of the application
        input_queue = asyncio.Queue()
        scope.setdefault("asgi", {"version": "3.0"})
        application_instance = self.application(
            scope=scope,
            receive=input_queue.get,
            send=partial(self.handle_reply, protocol),
        )
        # Run it, and stash the future for later checking
        if protocol not in self.connections:
            return None
        self.connections[protocol]["application_instance"] = asyncio.ensure_future(
            application_instance,
            loop=asyncio.get_event_loop(),
        )
        return input_queue

    async def handle_reply(self, protocol, message):
        """
        Coroutine that jumps the reply message from asyncio to Twisted
        """
        # Don't do anything if the connection is closed or does not exist
        if protocol not in self.connections or self.connections[protocol].get(
            "disconnected", None
        ):
            return
        try:
            self.check_headers_type(message)
        except ValueError:
            # Ensure to send SOME reply.
            protocol.basic_error(500, b"Server Error", "Server Error")
            raise
        # Let the protocol handle it
        protocol.handle_reply(message)

    @staticmethod
    def check_headers_type(message):
        if not message["type"] == "http.response.start":
            return
        for k, v in message.get("headers", []):
            if not isinstance(k, bytes):
                raise ValueError(
                    "Header name '{}' expected to be `bytes`, but got `{}`".format(
                        k, type(k)
                    )
                )
            if not isinstance(v, bytes):
                raise ValueError(
                    "Header value '{}' expected to be `bytes`, but got `{}`".format(
                        v, type(v)
                    )
                )

    ### Utility

    def application_checker(self):
        """
        Goes through the set of current application Futures and cleans up
        any that are done/prints exceptions for any that errored.
        """
        for protocol, details in list(self.connections.items()):
            disconnected = details.get("disconnected", None)
            application_instance = details.get("application_instance", None)
            # First, see if the protocol disconnected and the app has taken
            # too long to close up
            if (
                disconnected
                and time.time() - disconnected > self.application_close_timeout
            ):
                if application_instance and not application_instance.done():
                    logger.warning(
                        "Application instance %r for connection %s took too long to shut down and was killed.",
                        application_instance,
                        repr(protocol),
                    )
                    application_instance.cancel()
            # Then see if the app is done and we should reap it
            if application_instance and application_instance.done():
                try:
                    exception = application_instance.exception()
                except (CancelledError, asyncio.CancelledError):
                    # Future cancellation. We can ignore this.
                    pass
                else:
                    if exception:
                        if isinstance(exception, KeyboardInterrupt):
                            # Protocol is asking the server to exit (likely during test)
                            self.stop()
                        else:
                            logger.error(
                                "Exception inside application: %s",
                                exception,
                                exc_info=exception,
                            )
                            if not disconnected:
                                protocol.handle_exception(exception)
                del self.connections[protocol]["application_instance"]
                application_instance = None
                # The application task has truly ended: release its task
                # share now. The network share (if any) was already released
                # by protocol_disconnected, so the work budget is not freed
                # early while the task is still alive.
                ticket = self._admission_tickets.get(protocol)
                if ticket is not None and self.admission is not None:
                    if ticket.kind == KIND_HTTP:
                        self.admission.release_task(ticket)
            # Check to see if protocol is closed and app is closed so we can remove it
            if not application_instance and disconnected:
                # Safety net: free anything still booked (covers WebSocket
                # tickets and unusual terminal races). Idempotent.
                self.release_admission_ticket(protocol)
                del self.connections[protocol]
        if self._serving:
            self._application_checker_call = reactor.callLater(
                1, self.application_checker
            )

    def kill_all_applications(self):
        """
        Kills all application coroutines before reactor exit.
        """
        # Stop admitting new work and wake every queued waiter with a
        # shutdown rejection. Already granted tickets are left for the
        # normal cancel/reap flow below; the existing shutdown sequence is
        # otherwise unchanged.
        if self.admission is not None:
            self.admission.close()
        # Send cancel to all coroutines
        wait_for = []
        for details in self.connections.values():
            application_instance = details.get("application_instance")
            if application_instance is not None and not application_instance.done():
                application_instance.cancel()
                wait_for.append(application_instance)
        logger.info("Killed %i pending application instances", len(wait_for))
        # Make Twisted wait until they're all dead
        wait_deferred = defer.Deferred.fromFuture(asyncio.gather(*wait_for))
        wait_deferred.addErrback(lambda x: None)
        return wait_deferred

    def timeout_checker(self):
        """
        Called periodically to enforce timeout rules on all connections.
        Also checks pings at the same time.
        """
        for protocol in list(self.connections.keys()):
            protocol.check_timeouts()
        if self._serving:
            self._timeout_checker_call = reactor.callLater(
                2, self.timeout_checker
            )

    def log_action(self, protocol, action, details):
        """
        Dispatches to any registered action logger, if there is one.
        """
        if self.action_logger:
            self.action_logger(protocol, action, details)
