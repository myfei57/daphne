import logging
import multiprocessing
import os
import pickle
import re
import tempfile
import traceback
from concurrent.futures import CancelledError


class BaseDaphneTestingInstance:
    """
    Launches an instance of Daphne in a subprocess, with a host and port
    attribute allowing you to call it.

    Works as a context manager.
    """

    startup_timeout = 2

    def __init__(
        self,
        xff=False,
        http_timeout=None,
        request_buffer_size=None,
        websocket_max_message_size=None,
        websocket_max_frame_size=None,
        server_certfile=None,
        server_keyfile=None,
        client_cafile=None,
        *,
        application,
    ):
        self.xff = xff
        self.http_timeout = http_timeout
        self.host = "127.0.0.1"
        self.request_buffer_size = request_buffer_size
        self.websocket_max_message_size = websocket_max_message_size
        self.websocket_max_frame_size = websocket_max_frame_size
        # When server_certfile is set the child process serves TLS using the
        # PEM files on disk.  client_cafile additionally requires and verifies
        # a client certificate (mutual TLS).
        self.server_certfile = server_certfile
        self.server_keyfile = server_keyfile
        self.client_cafile = client_cafile
        self.application = application

    def get_application(self):
        return self.application

    def __enter__(self):
        # Option Daphne features
        kwargs = {}
        if self.request_buffer_size:
            kwargs["request_buffer_size"] = self.request_buffer_size
        # Optionally enable X-Forwarded-For support.
        if self.xff:
            kwargs["proxy_forwarded_address_header"] = "X-Forwarded-For"
            kwargs["proxy_forwarded_port_header"] = "X-Forwarded-Port"
            kwargs["proxy_forwarded_proto_header"] = "X-Forwarded-Proto"
        if self.http_timeout:
            kwargs["http_timeout"] = self.http_timeout
        if self.websocket_max_message_size is not None:
            kwargs["websocket_max_message_size"] = self.websocket_max_message_size
        if self.websocket_max_frame_size is not None:
            kwargs["websocket_max_frame_size"] = self.websocket_max_frame_size
        # Start up process
        self.process = DaphneProcess(
            host=self.host,
            get_application=self.get_application,
            kwargs=kwargs,
            setup=self.process_setup,
            teardown=self.process_teardown,
            server_certfile=self.server_certfile,
            server_keyfile=self.server_keyfile,
            client_cafile=self.client_cafile,
        )
        self.process.start()
        # Wait for the port
        if self.process.ready.wait(self.startup_timeout):
            self.port = self.process.port.value
            return self
        else:
            if self.process.errors.empty():
                raise RuntimeError("Daphne did not start up, no error caught")
            else:
                error, traceback = self.process.errors.get(False)
                raise RuntimeError("Daphne did not start up:\n%s" % traceback)

    def __exit__(self, exc_type, exc_value, traceback):
        # Shut down the process
        self.process.terminate()
        del self.process

    def process_setup(self):
        """
        Called by the process just before it starts serving.
        """
        pass

    def process_teardown(self):
        """
        Called by the process just after it stops serving
        """
        pass

    def get_received(self):
        pass


class DaphneTestingInstance(BaseDaphneTestingInstance):
    def __init__(self, *args, **kwargs):
        self.lock = multiprocessing.Lock()
        super().__init__(*args, **kwargs, application=TestApplication(lock=self.lock))

    def __enter__(self):
        # Clear result storage
        TestApplication.delete_setup()
        TestApplication.delete_result()
        return super().__enter__()

    def get_received(self):
        """
        Returns the scope and messages the test application has received
        so far. Note you'll get all messages since scope start, not just any
        new ones since the last call.

        Also checks for any exceptions in the application. If there are,
        raises them.
        """
        try:
            with self.lock:
                inner_result = TestApplication.load_result()
        except FileNotFoundError:
            raise ValueError("No results available yet.")
        # Check for exception
        if "exception" in inner_result:
            raise inner_result["exception"]
        return inner_result["scope"], inner_result["messages"]

    def add_send_messages(self, messages):
        """
        Adds messages for the application to send back.
        The next time it receives an incoming message, it will reply with these.
        """
        TestApplication.save_setup(response_messages=messages)


class ScopeRecorderApplication:
    """
    ASGI application that persists every connection scope into its own
    pickle file (one per request path) instead of a single shared slot.

    This makes it possible to inspect the independent, immutable scopes of
    several concurrent or keep-alive connections, which the single-slot
    :class:`TestApplication` storage cannot keep apart.  Only the scope is
    recorded; the application replies with a minimal successful HTTP or
    WebSocket response.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir

    @staticmethod
    def _key_for_path(path):
        # Turn a path such as "/conn/3" into a safe file name component.
        return re.sub(r"[^A-Za-z0-9_.-]", "_", path.strip("/")) or "root"

    def _record(self, scope):
        key = self._key_for_path(scope["path"])
        with open(os.path.join(self.output_dir, "scope-%s.pkl" % key), "wb") as fh:
            pickle.dump(scope, fh)

    async def __call__(self, scope, receive, send):
        self._record(scope)
        if scope["type"] == "http":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"OK"})
            # Stay alive until Daphne signals the end of the request.
            while True:
                if (await receive())["type"] == "http.disconnect":
                    break
        else:
            # Complete the WebSocket handshake then wait for disconnect.
            await receive()
            await send({"type": "websocket.accept"})
            while True:
                if (await receive())["type"] == "websocket.disconnect":
                    break


class ScopeRecorderTestingInstance(BaseDaphneTestingInstance):
    """
    Testing instance backed by :class:`ScopeRecorderApplication`.

    Each connection scope is written separately under ``output_dir`` and can
    be read back with :meth:`get_scope`.  All constructor keyword arguments
    supported by :class:`BaseDaphneTestingInstance` (including the TLS PEM
    file options) are passed through.
    """

    def __init__(self, output_dir, *args, **kwargs):
        self.output_dir = output_dir
        super().__init__(
            *args, application=ScopeRecorderApplication(output_dir), **kwargs
        )

    def scope_path(self, path):
        key = ScopeRecorderApplication._key_for_path(path)
        return os.path.join(self.output_dir, "scope-%s.pkl" % key)

    def get_scope(self, path):
        with open(self.scope_path(path), "rb") as fh:
            return pickle.load(fh)

    def has_scope(self, path):
        return os.path.exists(self.scope_path(path))


class DaphneProcess(multiprocessing.Process):
    """
    Process subclass that launches and runs a Daphne instance, communicating the
    port it ends up listening on back to the parent process.
    """

    def __init__(
        self,
        host,
        get_application,
        kwargs=None,
        setup=None,
        teardown=None,
        port=None,
        server_certfile=None,
        server_keyfile=None,
        client_cafile=None,
    ):
        super().__init__()
        self.host = host
        self.get_application = get_application
        self.kwargs = kwargs or {}
        self.setup = setup
        self.teardown = teardown
        self.port = multiprocessing.Value("i", port if port is not None else 0)
        self.ready = multiprocessing.Event()
        self.errors = multiprocessing.Queue()
        self.server_certfile = server_certfile
        self.server_keyfile = server_keyfile
        self.client_cafile = client_cafile

    def _build_tls_endpoint(self, reactor):
        """
        Build a programmatic TLS server endpoint from PEM files on disk.

        When ``client_cafile`` is provided the endpoint requires and verifies
        a client certificate against that CA (mutual TLS); otherwise the
        server presents its own certificate but does not request one.
        """
        from twisted.internet import ssl as tssl
        from twisted.internet.endpoints import SSL4ServerEndpoint

        with open(self.server_certfile, "rb") as cert_fh:
            cert_pem = cert_fh.read()
        with open(self.server_keyfile, "rb") as key_fh:
            key_pem = key_fh.read()
        private_certificate = tssl.PrivateCertificate.loadPEM(
            cert_pem + b"\n" + key_pem
        )
        options_kwargs = {
            "privateKey": private_certificate.privateKey.original,
            "certificate": private_certificate.original,
        }
        if self.client_cafile is not None:
            # Specifying trustRoot both requires a client certificate and
            # verifies it against the given CA during the handshake.
            with open(self.client_cafile, "rb") as ca_fh:
                options_kwargs["trustRoot"] = tssl.Certificate.loadPEM(ca_fh.read())
        context_factory = tssl.CertificateOptions(**options_kwargs)
        return SSL4ServerEndpoint(
            reactor,
            self.port.value,
            context_factory,
            interface=self.host,
        )

    def run(self):
        # OK, now we are in a forked child process, and want to use the reactor.
        # However, FreeBSD systems like MacOS do not fork the underlying Kqueue,
        # which asyncio (hence asyncioreactor) is built on.
        # Therefore, we should uninstall the broken reactor and install a new one.
        _reinstall_reactor()

        from twisted.internet import reactor

        from .endpoints import build_endpoint_description_strings
        from .server import Server

        application = self.get_application()

        try:
            # Create the server class
            if self.server_certfile is not None:
                # TLS (optionally mutual TLS) via a programmatic endpoint.
                endpoints = [self._build_tls_endpoint(reactor)]
            else:
                endpoints = build_endpoint_description_strings(
                    host=self.host, port=self.port.value
                )
            self.server = Server(
                application=application,
                endpoints=endpoints,
                signal_handlers=False,
                **self.kwargs,
            )
            # Set up a poller to look for the port
            reactor.callLater(0.1, self.resolve_port)
            # Run with setup/teardown
            if self.setup is not None:
                self.setup()
            try:
                self.server.run()
            finally:
                if self.teardown is not None:
                    self.teardown()
        except BaseException as e:
            # Put the error on our queue so the parent gets it
            self.errors.put((e, traceback.format_exc()))

    def resolve_port(self):
        from twisted.internet import reactor

        if self.server.listening_addresses:
            self.port.value = self.server.listening_addresses[0][1]
            self.ready.set()
        else:
            reactor.callLater(0.1, self.resolve_port)


class TestApplication:
    """
    An application that receives one or more messages, sends a response,
    and then quits the server. For testing.
    """

    setup_storage = os.path.join(tempfile.gettempdir(), "setup.testio")
    result_storage = os.path.join(tempfile.gettempdir(), "result.testio")

    def __init__(self, lock):
        self.lock = lock
        self.messages = []

    async def __call__(self, scope, receive, send):
        self.scope = scope
        # Receive input and send output
        logging.debug("test app coroutine alive")
        try:
            while True:
                # Receive a message and save it into the result store
                self.messages.append(await receive())
                self.lock.acquire()
                logging.debug("test app received %r", self.messages[-1])
                self.save_result(self.scope, self.messages)
                self.lock.release()
                # See if there are any messages to send back
                setup = self.load_setup()
                self.delete_setup()
                for message in setup["response_messages"]:
                    await send(message)
                    logging.debug("test app sent %r", message)
        except Exception as e:
            if isinstance(e, CancelledError):
                # Don't catch task-cancelled errors!
                raise
            else:
                self.save_exception(e)

    @classmethod
    def save_setup(cls, response_messages):
        """
        Stores setup information.
        """
        with open(cls.setup_storage, "wb") as fh:
            pickle.dump({"response_messages": response_messages}, fh)

    @classmethod
    def load_setup(cls):
        """
        Returns setup details.
        """
        try:
            with open(cls.setup_storage, "rb") as fh:
                return pickle.load(fh)
        except FileNotFoundError:
            return {"response_messages": []}

    @classmethod
    def save_result(cls, scope, messages):
        """
        Saves details of what happened to the result storage.
        We could use pickle here, but that seems wrong, still, somehow.
        """
        with open(cls.result_storage, "wb") as fh:
            pickle.dump({"scope": scope, "messages": messages}, fh)

    @classmethod
    def save_exception(cls, exception):
        """
        Saves details of what happened to the result storage.
        We could use pickle here, but that seems wrong, still, somehow.
        """
        with open(cls.result_storage, "wb") as fh:
            pickle.dump({"exception": exception}, fh)

    @classmethod
    def load_result(cls):
        """
        Returns result details.
        """
        with open(cls.result_storage, "rb") as fh:
            return pickle.load(fh)

    @classmethod
    def delete_setup(cls):
        """
        Clears setup storage files.
        """
        try:
            os.unlink(cls.setup_storage)
        except OSError:
            pass

    @classmethod
    def delete_result(cls):
        """
        Clears result storage files.
        """
        try:
            os.unlink(cls.result_storage)
        except OSError:
            pass


def _reinstall_reactor():
    import asyncio
    import sys

    from twisted.internet import asyncioreactor

    # Uninstall the reactor.
    if "twisted.internet.reactor" in sys.modules:
        del sys.modules["twisted.internet.reactor"]

    # The daphne.server module may have already installed the reactor.
    # If so, using this module will use uninstalled one, thus we should
    # reimport this module too.
    if "daphne.server" in sys.modules:
        del sys.modules["daphne.server"]

    event_loop = asyncio.new_event_loop()
    asyncioreactor.install(event_loop)
    asyncio.set_event_loop(event_loop)
