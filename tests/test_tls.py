"""Real TLS handshakes with the certs CHARM generates, using the settings its servers run with."""
import socket
import ssl
import threading
from pathlib import Path

import pytest
from cryptography import x509

import charm_guard as g


def _serve(kwargs: dict, results: list):
    """A one-connection echo server configured from guard.tls_server_kwargs."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(kwargs["ssl_certfile"], kwargs["ssl_keyfile"])
    context.load_verify_locations(kwargs["ssl_ca_certs"])
    context.verify_mode = kwargs["ssl_cert_reqs"]
    listener = socket.create_server(("127.0.0.1", 0))

    def run():
        conn, _ = listener.accept()
        try:
            with context.wrap_socket(conn, server_side=True) as tls:
                results.append(tls.getpeercert())
                tls.sendall(tls.recv(16))
        except (ssl.SSLError, OSError) as e:
            results.append(e)
        finally:
            listener.close()

    threading.Thread(target=run, daemon=True).start()
    return listener.getsockname()[1]


def _exchange(port: int, context: ssl.SSLContext, host: str = "localhost") -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with context.wrap_socket(raw, server_hostname=host) as tls:
            tls.sendall(b"ping")
            return tls.recv(16)


@pytest.fixture
def foreign_ca(tmp_path, monkeypatch):
    """Certs with the right names from a different authority, then the real TLS_DIR again."""
    real = g.TLS_DIR
    monkeypatch.setattr(g, "TLS_DIR", tmp_path)
    g.generate_tls(["ChatAgent", "EmailAgent"])
    client = ssl.create_default_context(cafile=str(tmp_path / "ca.crt"))
    client.load_cert_chain(str(tmp_path / "ChatAgent.crt"), str(tmp_path / "ChatAgent.key"))
    server = g.tls_server_kwargs("EmailAgent")
    monkeypatch.setattr(g, "TLS_DIR", real)
    return client, server


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_agents_with_charm_certs_connect_encrypted(host):
    results = []
    port = _serve(g.tls_server_kwargs("EmailAgent"), results)
    assert _exchange(port, g.tls_context(g.Identity("ChatAgent")), host) == b"ping"
    assert dict(x[0] for x in results[0]["subject"])["commonName"] == "ChatAgent"


def test_client_without_a_cert_is_refused():
    results = []
    port = _serve(g.tls_server_kwargs("EmailAgent"), results)
    no_cert = ssl.create_default_context(cafile=str(g.TLS_DIR / "ca.crt"))
    with pytest.raises((ssl.SSLError, OSError)):
        assert _exchange(port, no_cert) != b"ping" or pytest.fail("server echoed to a client with no cert")


def test_client_cert_from_another_authority_is_refused(foreign_ca):
    client, _ = foreign_ca
    # Trust the real server, but present a cert the real CA never signed.
    client.load_verify_locations(str(g.TLS_DIR / "ca.crt"))
    results = []
    port = _serve(g.tls_server_kwargs("EmailAgent"), results)
    with pytest.raises((ssl.SSLError, OSError)):
        assert _exchange(port, client) != b"ping" or pytest.fail("server echoed to a foreign cert")


def test_client_refuses_a_server_from_another_authority(foreign_ca):
    _, server = foreign_ca
    results = []
    port = _serve(server, results)
    with pytest.raises(ssl.SSLCertVerificationError):
        _exchange(port, g.tls_context(g.Identity("ChatAgent")))


def test_client_refuses_a_server_cert_for_another_hostname():
    results = []
    port = _serve(g.tls_server_kwargs("EmailAgent"), results)
    with pytest.raises(ssl.SSLCertVerificationError):
        _exchange(port, g.tls_context(g.Identity("ChatAgent")), "evil.test")


def test_server_requires_client_certs():
    assert g.tls_server_kwargs("EmailAgent")["ssl_cert_reqs"] == ssl.CERT_REQUIRED
    assert "--ssl-cert-reqs" in g.tls_server_args("EmailAgent")


def test_private_keys_are_owner_only():
    for name in ("ca", "EmailAgent"):
        assert (g.TLS_DIR / f"{name}.key").stat().st_mode & 0o077 == 0


def test_certs_name_the_principal_and_only_loopback():
    cert = x509.load_pem_x509_certificate(Path(g.TLS_DIR / "EmailAgent.crt").read_bytes())
    names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert names.get_values_for_type(x509.DNSName) == ["localhost"]
    assert [str(i) for i in names.get_values_for_type(x509.IPAddress)] == ["127.0.0.1"]
    assert cert.subject.rfc4514_string() == "CN=EmailAgent"


def test_generate_tls_keeps_existing_certs():
    before = (g.TLS_DIR / "EmailAgent.crt").read_bytes()
    assert g.generate_tls() == []
    assert (g.TLS_DIR / "EmailAgent.crt").read_bytes() == before
