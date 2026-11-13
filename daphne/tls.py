"""
ASGI TLS extension (``0.3``) transport evidence adapter.

This module turns the *real* TLS state of a Twisted connection into the
immutable ``extensions["tls"]`` dictionary described by the ASGI TLS
extension specification version ``0.3``::

    {"spec_version": "0.3", "client_cert_chain": [PEM, PEM, ...]}

Security and correctness rules implemented here:

* The extension is derived solely from the connection transport.  HTTP
  headers (including ``X-Forwarded-Proto``) and the request ``scheme`` are
  never consulted, so a plaintext connection cannot forge TLS evidence.
* Only certificates presented by the peer during the TLS handshake are
  reported.  The server's own certificate, the configured trust store, and
  strings received in HTTP headers can never end up in
  ``client_cert_chain``.
* The chain is ordered leaf-first, i.e. the client certificate is first and
  every subsequent certificate signed the preceding one.  Missing issuers
  are never fabricated: if the peer only sent a leaf, only the leaf is
  returned.
* The OpenSSL connection is read after the handshake has completed; this
  code never changes verification policy, so a failure here can never weaken
  TLS authentication.  The mere presence of a certificate does not imply any
  application-level authorization.
* Only PEM text (immutable ``str``) is handed to the application.  Mutable
  OpenSSL objects are converted immediately and never stored on a scope.
* Chain length, per-certificate and total PEM size, and extraction time are
  bounded.  If a bound is exceeded or the transport cannot be read, the
  extension is still reported (the connection really is TLS) but with an
  empty chain rather than partial or oversized data.

Importing this module does not touch the reactor and never opens a
connection.
"""

import logging
import time

from OpenSSL import crypto
from twisted.internet.interfaces import ISSLTransport
from twisted.protocols.policies import ProtocolWrapper
from twisted.protocols.tls import TLSMemoryBIOProtocol

logger = logging.getLogger(__name__)

#: ASGI TLS extension version implemented by this module.
TLS_SPEC_VERSION = "0.3"

#: Maximum number of certificates allowed in ``client_cert_chain``.  Real
#: world chains are only a few certificates long; the bound is defensive.
MAX_CLIENT_CERT_CHAIN_LENGTH = 10

#: Maximum size, in bytes, of a single PEM-encoded certificate.  A typical
#: certificate PEM is 1-2 KiB, so 64 KiB is comfortably above normal traffic.
MAX_SINGLE_CERT_PEM_BYTES = 64 * 1024

#: Maximum total size, in bytes, of every PEM certificate in the chain.
MAX_TOTAL_PEM_BYTES = 256 * 1024

#: Wall-clock budget for extracting and encoding the whole chain.  Encoding a
#: handful of certificates normally takes well under a millisecond.
MAX_EXTRACTION_TIME = 0.1

#: Maximum number of ``ProtocolWrapper`` layers traversed while looking for
#: the TLS transport.
MAX_TRANSPORT_UNWRAP_DEPTH = 8


def _empty_extension():
    """
    Build a fresh, valid TLS extension carrying no certificates.
    """
    return {"spec_version": TLS_SPEC_VERSION, "client_cert_chain": []}


def _find_tls_transport(transport):
    """
    Locate the native Twisted TLS transport for a connection.

    The application protocol's transport may be the TLS wrapper directly or
    may be separated from it by one or more :class:`ProtocolWrapper` layers.
    Unwrap explicitly (via ``transport``, which points at the lower layer)
    until a native TLS transport is found or the stack is exhausted.

    :return: the transport providing TLS evidence, or ``None`` when the
        connection is not TLS (or TLS cannot be safely identified).
    """
    current = transport
    for _ in range(MAX_TRANSPORT_UNWRAP_DEPTH + 1):
        if current is None:
            return None
        # Twisted's native memory-BIO TLS implementation (and subclasses such
        # as BufferingTLSTransport) is a ProtocolWrapper exposing the OpenSSL
        # connection through getHandle().
        if isinstance(current, TLSMemoryBIOProtocol):
            return current
        # Backstop for alternative ISSLTransport implementations.
        if ISSLTransport(current, None) is not None and hasattr(current, "getHandle"):
            return current
        if isinstance(current, ProtocolWrapper):
            # Descend to the wrapped lower-level transport.  ProtocolWrapper
            # delegates unknown attributes, but ``transport`` is a real
            # attribute set by makeConnection().
            current = getattr(current, "transport", None)
            continue
        # Reached a plain (e.g. raw TCP) transport: this is not TLS.
        return None
    logger.warning(
        "TLS transport unwrapping exceeded the depth limit (%d); "
        "no TLS extension will be reported for this connection",
        MAX_TRANSPORT_UNWRAP_DEPTH,
    )
    return None


def _peer_certificates(connection):
    """
    Return the client certificates presented during the handshake, ordered
    leaf-first.

    On the server side PyOpenSSL exposes the leaf separately from the rest of
    the chain: ``get_peer_certificate()`` returns the leaf while
    ``get_peer_cert_chain()`` returns the intermediates presented by the
    peer (never the trust-store root).  Recombining them yields the exact
    leaf-to-issuer order required by the ASGI spec.

    :return: a list of OpenSSL ``X509`` objects, possibly empty.
    """
    leaf = connection.get_peer_certificate()
    if leaf is None:
        # No client certificate was presented.  Never fabricate one.
        return []
    presented_chain = connection.get_peer_cert_chain()
    if presented_chain:
        return [leaf, *presented_chain]
    return [leaf]


def _encode_chain(certificates):
    """
    Convert OpenSSL certificates into bounded, immutable PEM strings.

    :raises ValueError: if a configured limit would be exceeded.
    """
    if len(certificates) > MAX_CLIENT_CERT_CHAIN_LENGTH:
        raise ValueError(
            "client certificate chain length %d exceeds limit %d"
            % (len(certificates), MAX_CLIENT_CERT_CHAIN_LENGTH)
        )

    pems = []
    total_bytes = 0
    started = time.monotonic()
    for certificate in certificates:
        if time.monotonic() - started > MAX_EXTRACTION_TIME:
            raise ValueError("client certificate extraction exceeded the time limit")
        pem_bytes = crypto.dump_certificate(crypto.FILETYPE_PEM, certificate)
        if len(pem_bytes) > MAX_SINGLE_CERT_PEM_BYTES:
            raise ValueError("a client certificate PEM exceeded the size limit")
        total_bytes += len(pem_bytes)
        if total_bytes > MAX_TOTAL_PEM_BYTES:
            raise ValueError("the client certificate chain PEM exceeded the size limit")
        # PEM is ASCII text; decode to an immutable unicode string for the app.
        pems.append(pem_bytes.decode("ascii"))
    return pems


def build_tls_extension(transport):
    """
    Build the ASGI ``0.3`` TLS extension for ``transport``.

    :param transport: the transport of the HTTP or WebSocket protocol
        handling the current connection/request.
    :return: a fresh ``{"spec_version": ..., "client_cert_chain": [...]}``
        dictionary for a genuine TLS connection, or ``None`` for a plaintext
        connection (in which case the ASGI server must not provide the
        extension at all).
    """
    tls_transport = _find_tls_transport(transport)
    if tls_transport is None:
        return None

    try:
        connection = tls_transport.getHandle()
        if connection is None:
            # The TLS connection has already been torn down; report TLS with
            # no chain rather than reaching into freed state.
            return _empty_extension()
        certificates = _peer_certificates(connection)
        return {
            "spec_version": TLS_SPEC_VERSION,
            "client_cert_chain": _encode_chain(certificates),
        }
    except Exception as exc:
        # Extraction is purely diagnostic.  TLS authentication already
        # happened during the handshake, so a failure here must not change
        # connection handling.  Keep the TLS capability signal but do not
        # leak partial/certificate data.  Never log PEM content or keys.
        logger.warning(
            "Unable to extract the client certificate chain (%s); "
            "reporting an empty client_cert_chain",
            type(exc).__name__,
        )
        return _empty_extension()
