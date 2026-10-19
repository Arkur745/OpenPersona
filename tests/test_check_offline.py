"""Tests for scripts/check_offline.py's network guard.

The guard is what makes the offline check meaningful: if it let a
non-localhost connection through, every PASS it prints would be
worthless. The backend checks themselves need downloaded models and a
running Ollama, so they are run by hand (see README.md), not here.
"""

from __future__ import annotations

import importlib.util
import socket
import threading
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_offline.py"
_spec = importlib.util.spec_from_file_location("check_offline", _SCRIPT)
check_offline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_offline)


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.5.5.5", "::1", "[::1]", "localhost", "LOCALHOST.", "ollama.localhost",
     None, b"127.0.0.1"],
)
def test_loopback_hosts_are_allowed(host):
    assert check_offline.is_loopback(host)


@pytest.mark.parametrize(
    "host", ["huggingface.co", "8.8.8.8", "192.168.1.10", "2001:4860:4860::8888", "localhost.evil.com"]
)
def test_other_hosts_are_not_loopback(host):
    assert not check_offline.is_loopback(host)


@pytest.fixture
def guard():
    check_offline.blocked_attempts.clear()
    uninstall = check_offline.install_network_guard()
    yield check_offline.blocked_attempts
    uninstall()


def test_guard_blocks_dns_and_connect_to_external_hosts(guard):
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("huggingface.co", 443)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(OSError):
        s.connect(("93.184.216.34", 80))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(OSError):
        s.connect_ex(("93.184.216.34", 80))
    assert "dns huggingface.co" in guard
    assert "connect 93.184.216.34:80" in guard


def test_guard_still_allows_localhost_connections(guard):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = threading.Thread(target=lambda: server.accept()[0].close())
    accepted.start()
    try:
        with socket.create_connection(("localhost", port), timeout=5):
            pass
    finally:
        accepted.join(timeout=5)
        server.close()
    assert guard == []


def test_uninstall_restores_the_real_socket_functions():
    real = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)
    check_offline.install_network_guard()()
    assert (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo) == real
