"""
Tests for the ASGI TLS extension (spec version ``0.3``).

These tests build a temporary certificate authority, serve Daphne over real
TLS (optionally mutual TLS) in a subprocess, and connect with real TLS
clients.  The PEM certificates exposed in the connection scope are parsed
back into x509 objects so their fingerprints and leaf-to-issuer ordering can
be asserted against the certificates that were actually provisioned.
"""

import http.client
import os
import shutil
import socket
import ssl
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from OpenSSL import crypto
from twisted.internet.interfaces import ISSLTransport
from twisted.internet.protocol import Protocol
from twisted.protocols.policies import ProtocolWrapper, WrappingFactory
from zope.interface import directlyProvides

from daphne import tls as tls_module
from daphne.testing import ScopeRecorderTestingInstance
from daphne.tls import TLS_SPEC_VERSION, build_tls_extension


def _now():
    return datetime.now(timezone.utc)


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def make_ca(cn, issuer_cert=None, issuer_key=None, path_length=None):
    key = ec.generate_private_key(ec.SECP256R1())
    subject = _name(cn)
    issuer_name = subject if issuer_cert is None else issuer_cert.subject
    signing_key = key if issuer_key is None else issuer_key
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - timedelta(days=1))
        .not_valid_after(_now() + timedelta(days=365))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=path_length), critical=True
        )
        .sign(signing_key, hashes.SHA256())
    )
    return key, cert


def make_leaf(cn, issuer_cert, issuer_key, eku):
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(issuer_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - timedelta(days=1))
        .not_valid_after(_now() + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([eku]), critical=True)
        .sign(issuer_key, hashes.SHA256())
    )
    return key, cert


def cert_pem(cert):
    return cert.public_bytes(serialization.Encoding.PEM)


def key_pem(key):
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def fingerprint(cert):
    return cert.fingerprint(hashes.SHA256()).hex()


class TemporaryPKI:
    """Generates and persists a root -> intermediate -> leaf hierarchy."""

    def __init__(self, directory):
        self.directory = directory
        self.root_key, self.root = make_ca("Test Root CA", path_length=3)
        self.intermediate_key, self.intermediate = make_ca(
            "Test Intermediate CA", self.root, self.root_key, path_length=1
        )
        self.server_key, self.server = make_leaf(
            "server-leaf", self.root, self.root_key, ExtendedKeyUsageOID.SERVER_AUTH
        )
        # A default client leaf signed by the intermediate.
        self.client_key, self.client = make_leaf(
            "client-leaf",
            self.intermediate,
            self.intermediate_key,
            ExtendedKeyUsageOID.CLIENT_AUTH,
        )
        self._write("root.pem", cert_pem(self.root))
        self._write("server.pem", cert_pem(self.server))
        self._write("server_key.pem", key_pem(self.server_key))
        self._write("intermediate.pem", cert_pem(self.intermediate))
        self._write_client("client", self.client, self.client_key)

    def _write(self, name, data):
        with open(os.path.join(self.directory, name), "wb") as fh:
            fh.write(data)

    def _write_client(self, label, client_cert, client_key):
        self._write("%s.pem" % label, cert_pem(client_cert))
        self._write("%s_key.pem" % label, key_pem(client_key))
        # The chain presented by the client: leaf followed by its intermediate
        # (never the trust-store root).
        self._write(
            "%s_chain.pem" % label,
            cert_pem(client_cert) + cert_pem(self.intermediate),
        )

    def add_client(self, cn):
        """Create an additional distinct client leaf for concurrency tests."""
        label = cn.replace(" ", "_")
        key, cert = make_leaf(
            cn,
            self.intermediate,
            self.intermediate_key,
            ExtendedKeyUsageOID.CLIENT_AUTH,
        )
        self._write_client(label, cert, key)
        return cert, {
            "chain": os.path.join(self.directory, "%s_chain.pem" % label),
            "key": os.path.join(self.directory, "%s_key.pem" % label),
        }

    @property
    def server_kwargs(self):
        return dict(
            server_certfile=os.path.join(self.directory, "server.pem"),
            server_keyfile=os.path.join(self.directory, "server_key.pem"),
        )

    @property
    def mutual_kwargs(self):
        kwargs = dict(self.server_kwargs)
        kwargs["client_cafile"] = os.path.join(self.directory, "root.pem")
        return kwargs

    def client_context(self, chain=None, key=None, verify_server=True):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED if verify_server else ssl.CERT_NONE
        context.load_verify_locations(cadata=cert_pem(self.root).decode("ascii"))
        if chain is not None:
            context.load_cert_chain(certfile=chain, keyfile=key)
        return context

    @property
    def default_client_context(self):
        return self.client_context(
            chain=os.path.join(self.directory, "client_chain.pem"),
            key=os.path.join(self.directory, "client_key.pem"),
        )


def parse_scope_chain(scope):
    """
    Returns (extension_dict, parsed_cryptography_certs) from a scope and
    validates the structural shape required by ASGI TLS 0.3.
    """
    extension = scope["extensions"]["tls"]
    # Only the 0.3 fields are permitted.
    assert set(extension.keys()) == {"spec_version", "client_cert_chain"}
    assert extension["spec_version"] == TLS_SPEC_VERSION
    pems = extension["client_cert_chain"]
    assert all(isinstance(pem, str) for pem in pems)
    certs = [x509.load_pem_x509_certificate(pem.encode("ascii")) for pem in pems]
    return extension, certs


def wait_for_scope(instance, path, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if instance.has_scope(path):
            return instance.get_scope(path)
        time.sleep(0.02)
    raise AssertionError("scope for %r was never recorded" % path)


def https_get(host, port, context, path):
    conn = http.client.HTTPSConnection(host, port, context=context, timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read()
    return conn, response.status, body


def wss_handshake(host, port, context, path):
    raw = socket.create_connection((host, port), timeout=5)
    sock = context.wrap_socket(raw, server_hostname=None)
    request = (
        "GET %s HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Origin: https://example.com\r\n\r\n" % path
    ).encode("ascii")
    sock.sendall(request)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return sock, data


class FakeConnection:
    def __init__(self, leaf, chain):
        self._leaf = leaf
        self._chain = chain

    def get_peer_certificate(self):
        return self._leaf

    def get_peer_cert_chain(self):
        return self._chain


class FakeTLSTransport:
    def __init__(self, connection):
        self._connection = connection

    def getHandle(self):
        return self._connection


def make_fake_tls(leaf, chain):
    transport = FakeTLSTransport(FakeConnection(leaf, chain))
    # Mimic Twisted's native TLS wrapper advertising ISSLTransport.
    directlyProvides(transport, ISSLTransport)
    return transport


def load_openssl_x509(pem_bytes):
    return crypto.load_certificate(crypto.FILETYPE_PEM, pem_bytes)


class TLSAdapterUnitTests(unittest.TestCase):
    """Deterministic unit tests that need no reactor or network."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.pki = TemporaryPKI(self.directory)
        self.leaf_x509 = load_openssl_x509(cert_pem(self.pki.client))
        self.intermediate_x509 = load_openssl_x509(cert_pem(self.pki.intermediate))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_plaintext_and_none_transport_have_no_extension(self):
        self.assertIsNone(build_tls_extension(None))
        # A bare object behaves like a raw TCP transport.
        self.assertIsNone(build_tls_extension(object()))

    def test_native_tls_transport_leaf_first_chain(self):
        transport = make_fake_tls(self.leaf_x509, [self.intermediate_x509])
        extension = build_tls_extension(transport)
        self.assertEqual(set(extension.keys()), {"spec_version", "client_cert_chain"})
        self.assertEqual(extension["spec_version"], "0.3")
        expected = [
            crypto.dump_certificate(crypto.FILETYPE_PEM, self.leaf_x509).decode(
                "ascii"
            ),
            crypto.dump_certificate(crypto.FILETYPE_PEM, self.intermediate_x509).decode(
                "ascii"
            ),
        ]
        self.assertEqual(extension["client_cert_chain"], expected)

    def test_chain_order_and_issuer_linkage(self):
        transport = make_fake_tls(self.leaf_x509, [self.intermediate_x509])
        _, certs = parse_scope_chain(
            {"extensions": {"tls": build_tls_extension(transport)}}
        )
        self.assertEqual(fingerprint(certs[0]), fingerprint(self.pki.client))
        self.assertEqual(fingerprint(certs[1]), fingerprint(self.pki.intermediate))
        # Each subsequent certificate signs the preceding one.
        self.assertEqual(certs[0].issuer, certs[1].subject)
        # The trust-store root and the server cert must never appear.
        chain_fingerprints = {fingerprint(c) for c in certs}
        self.assertNotIn(fingerprint(self.pki.root), chain_fingerprints)
        self.assertNotIn(fingerprint(self.pki.server), chain_fingerprints)

    def test_unwraps_protocol_wrapper_layers(self):
        tls_transport = make_fake_tls(self.leaf_x509, [self.intermediate_x509])
        wrapper = ProtocolWrapper(WrappingFactory(None), Protocol())
        # ProtocolWrapper.makeConnection would normally set this; emulate the
        # lower-level TLS transport the wrapper sits on top of.
        wrapper.transport = tls_transport
        extension = build_tls_extension(wrapper)
        self.assertIsNotNone(extension)
        _, certs = parse_scope_chain({"extensions": {"tls": extension}})
        self.assertEqual(fingerprint(certs[0]), fingerprint(self.pki.client))

    def test_no_client_certificate_is_empty_chain(self):
        transport = make_fake_tls(None, None)
        extension = build_tls_extension(transport)
        self.assertEqual(extension["spec_version"], "0.3")
        self.assertEqual(extension["client_cert_chain"], [])

    def test_leaf_only_does_not_fabricate_issuer(self):
        # Peer sent a leaf but no intermediates: report the leaf and nothing
        # else rather than inventing the missing issuer.
        transport = make_fake_tls(self.leaf_x509, None)
        extension = build_tls_extension(transport)
        self.assertEqual(len(extension["client_cert_chain"]), 1)

    def test_returns_fresh_immutable_strings(self):
        transport = make_fake_tls(self.leaf_x509, [self.intermediate_x509])
        first = build_tls_extension(transport)
        second = build_tls_extension(transport)
        self.assertIsNot(first, second)
        self.assertIsNot(first["client_cert_chain"], second["client_cert_chain"])
        self.assertEqual(first, second)
        for pem in first["client_cert_chain"]:
            self.assertIsInstance(pem, str)
        # No mutable OpenSSL object leaks into the scope.
        self.assertNotIn(
            type(self.leaf_x509), {type(pem) for pem in first["client_cert_chain"]}
        )

    def test_extraction_error_keeps_tls_extension_with_empty_chain(self):
        class BrokenTransport:
            def getHandle(self):
                raise RuntimeError("boom")

        transport = BrokenTransport()
        directlyProvides(transport, ISSLTransport)
        extension = build_tls_extension(transport)
        self.assertIsNotNone(extension)
        self.assertEqual(extension["client_cert_chain"], [])

    def test_missing_handle_after_teardown_is_empty_chain(self):
        transport = FakeTLSTransport(None)
        directlyProvides(transport, ISSLTransport)
        extension = build_tls_extension(transport)
        self.assertEqual(extension["client_cert_chain"], [])

    def test_chain_length_limit(self):
        with mock.patch.object(tls_module, "MAX_CLIENT_CERT_CHAIN_LENGTH", 1):
            transport = make_fake_tls(self.leaf_x509, [self.intermediate_x509])
            extension = build_tls_extension(transport)
        self.assertEqual(extension["client_cert_chain"], [])

    def test_single_cert_size_limit(self):
        with mock.patch.object(tls_module, "MAX_SINGLE_CERT_PEM_BYTES", 1):
            transport = make_fake_tls(self.leaf_x509, None)
            extension = build_tls_extension(transport)
        self.assertEqual(extension["client_cert_chain"], [])

    def test_total_size_limit(self):
        with mock.patch.object(tls_module, "MAX_TOTAL_PEM_BYTES", 1):
            transport = make_fake_tls(self.leaf_x509, None)
            extension = build_tls_extension(transport)
        self.assertEqual(extension["client_cert_chain"], [])

    def test_extraction_time_limit(self):
        with mock.patch.object(tls_module, "MAX_EXTRACTION_TIME", -1):
            transport = make_fake_tls(self.leaf_x509, None)
            extension = build_tls_extension(transport)
        self.assertEqual(extension["client_cert_chain"], [])


class TLSExtensionIntegrationTests(unittest.TestCase):
    """Real TLS/mTLS handshakes against a Daphne subprocess."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.pki = TemporaryPKI(self.directory)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_mtls_http_scope_reports_leaf_first_chain(self):
        with ScopeRecorderTestingInstance(
            self.directory, **self.pki.mutual_kwargs
        ) as server:
            conn, status, body = https_get(
                server.host, server.port, self.pki.default_client_context, "/mtls"
            )
            self.assertEqual(status, 200)
            conn.close()
            scope = wait_for_scope(server, "/mtls")

        self.assertEqual(scope["type"], "http")
        self.assertEqual(scope["scheme"], "https")
        extension, certs = parse_scope_chain(scope)
        self.assertEqual(
            [fingerprint(c) for c in certs],
            [fingerprint(self.pki.client), fingerprint(self.pki.intermediate)],
        )

    def test_tls_without_client_certificate_is_empty_chain(self):
        # No client_cafile: the server does not request a client certificate.
        with ScopeRecorderTestingInstance(
            self.directory, **self.pki.server_kwargs
        ) as server:
            context = self.pki.client_context()  # verifies server, no client cert
            conn, status, _ = https_get(server.host, server.port, context, "/nocert")
            self.assertEqual(status, 200)
            conn.close()
            scope = wait_for_scope(server, "/nocert")

        self.assertEqual(scope["scheme"], "https")
        extension, certs = parse_scope_chain(scope)
        self.assertEqual(certs, [])
        self.assertEqual(extension["client_cert_chain"], [])

    def test_untrusted_client_certificate_fails_handshake(self):
        # Reporting TLS evidence must never weaken TLS authentication: a
        # client certificate not chained to the configured trust root is
        # rejected during the handshake and never reaches an ASGI scope.
        rogue_key = ec.generate_private_key(ec.SECP256R1())
        rogue_cert = (
            x509.CertificateBuilder()
            .subject_name(_name("rogue-self-signed"))
            .issuer_name(_name("rogue-self-signed"))
            .public_key(rogue_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_now() - timedelta(days=1))
            .not_valid_after(_now() + timedelta(days=365))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None), critical=True
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=True,
            )
            .sign(rogue_key, hashes.SHA256())
        )
        chain_file = os.path.join(self.directory, "rogue_chain.pem")
        key_file = os.path.join(self.directory, "rogue_key.pem")
        with open(chain_file, "wb") as fh:
            fh.write(cert_pem(rogue_cert))
        with open(key_file, "wb") as fh:
            fh.write(key_pem(rogue_key))
        context = self.pki.client_context(chain=chain_file, key=key_file)
        # Pin TLS 1.2 so an unacceptable client certificate fails the
        # handshake deterministically rather than via a post-handshake alert.
        context.maximum_version = ssl.TLSVersion.TLSv1_2

        with ScopeRecorderTestingInstance(
            self.directory, **self.pki.mutual_kwargs
        ) as server:
            with self.assertRaises(ssl.SSLError):
                https_get(server.host, server.port, context, "/rogue")
            time.sleep(0.2)
            self.assertFalse(server.has_scope("/rogue"))

    def test_plaintext_with_forged_forwarded_proto_has_no_tls_extension(self):
        # Plaintext listener with proxy-header handling enabled.
        with ScopeRecorderTestingInstance(self.directory, xff=True) as server:
            conn = http.client.HTTPConnection(server.host, server.port, timeout=5)
            # The address header must be present for Daphne to consult the
            # proto header at all; send both to simulate a forged proxy.
            conn.request(
                "GET",
                "/forged",
                headers={
                    "X-Forwarded-For": "10.1.2.3",
                    "X-Forwarded-Proto": "https",
                },
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            conn.close()
            scope = wait_for_scope(server, "/forged")

        # The proxy header is allowed to change the scheme...
        self.assertEqual(scope["scheme"], "https")
        # ...but it must never fabricate TLS evidence on a plaintext socket.
        self.assertNotIn("extensions", scope)

    def test_wss_upgrade_preserves_tls_evidence(self):
        with ScopeRecorderTestingInstance(
            self.directory, **self.pki.mutual_kwargs
        ) as server:
            sock, raw = wss_handshake(
                server.host,
                server.port,
                self.pki.default_client_context,
                "/ws",
            )
            try:
                self.assertTrue(raw.startswith(b"HTTP/1.1 101"))
                scope = wait_for_scope(server, "/ws")
            finally:
                sock.close()

        self.assertEqual(scope["type"], "websocket")
        extension, certs = parse_scope_chain(scope)
        # The upgraded WebSocket keeps the identical connection certificate
        # chain as the underlying HTTPS connection.
        self.assertEqual(
            [fingerprint(c) for c in certs],
            [fingerprint(self.pki.client), fingerprint(self.pki.intermediate)],
        )

    def test_http_keepalive_requests_have_independent_scopes(self):
        with ScopeRecorderTestingInstance(
            self.directory, **self.pki.mutual_kwargs
        ) as server:
            conn = http.client.HTTPSConnection(
                server.host, server.port, context=self.pki.default_client_context
            )
            conn.request("GET", "/keep1")
            response_one = conn.getresponse()
            self.assertEqual(response_one.status, 200)
            response_one.read()
            conn.request("GET", "/keep2")
            response_two = conn.getresponse()
            self.assertEqual(response_two.status, 200)
            response_two.read()
            conn.close()
            scope_one = wait_for_scope(server, "/keep1")
            scope_two = wait_for_scope(server, "/keep2")

        ext_one, certs_one = parse_scope_chain(scope_one)
        ext_two, certs_two = parse_scope_chain(scope_two)
        # Same evidence values...
        self.assertEqual(ext_one["client_cert_chain"], ext_two["client_cert_chain"])
        # ...but independent, immutable per-request containers.
        self.assertIsNot(ext_one, ext_two)
        self.assertIsNot(ext_one["client_cert_chain"], ext_two["client_cert_chain"])
        ext_one["client_cert_chain"].append("tampered")
        self.assertNotIn("tampered", ext_two["client_cert_chain"])
        self.assertEqual(
            [fingerprint(c) for c in certs_one],
            [
                fingerprint(self.pki.client),
                fingerprint(self.pki.intermediate),
            ],
        )
        self.assertEqual(
            [fingerprint(c) for c in certs_two],
            [fingerprint(self.pki.client), fingerprint(self.pki.intermediate)],
        )

    def test_concurrent_connections_keep_distinct_evidence(self):
        client_specs = []
        expected = {}
        for index in range(4):
            cert, files = self.pki.add_client("client-%d" % index)
            context = self.pki.client_context(chain=files["chain"], key=files["key"])
            path = "/c%d" % index
            client_specs.append((path, context))
            expected[path] = fingerprint(cert)

        with ScopeRecorderTestingInstance(
            self.directory, **self.pki.mutual_kwargs
        ) as server:
            errors = []

            def worker(path, context):
                try:
                    conn, status, _ = https_get(server.host, server.port, context, path)
                    if status != 200:
                        errors.append("status %s for %s" % (status, path))
                    conn.close()
                except Exception as exc:  # pragma: no cover - failure reporting
                    errors.append("%s: %r" % (path, exc))

            threads = [
                threading.Thread(target=worker, args=spec) for spec in client_specs
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            self.assertEqual(errors, [])
            scopes = {path: wait_for_scope(server, path) for path, _ in client_specs}

        observed = {}
        for path, scope in scopes.items():
            _, certs = parse_scope_chain(scope)
            # Leaf is always first.
            observed[path] = fingerprint(certs[0])
            self.assertEqual(fingerprint(certs[-1]), fingerprint(self.pki.intermediate))
        # Every connection saw exactly its own certificate, never a neighbour's.
        self.assertEqual(observed, expected)
        self.assertEqual(len(set(observed.values())), len(expected))


if __name__ == "__main__":
    unittest.main()
