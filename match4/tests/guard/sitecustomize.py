"""Subprocess test guard: reject heavy imports and all non-loopback networking."""
import importlib.abc
import os
import socket
import sys


class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "torchvision"}:
            raise ImportError("heavy import forbidden in CPU tests: " + fullname)


sys.meta_path.insert(0, NoHeavyImports())
original_connect = socket.socket.connect
original_getaddrinfo = socket.getaddrinfo


def allowed(host):
    return host == "127.0.0.1" and os.environ.get("X052_TEST_NO_INET") != "1"


def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not allowed(address[0]):
        raise AssertionError("test attempted forbidden network: " + repr(address))
    return original_connect(self, address)


def getaddrinfo(host, *args, **kwargs):
    if not allowed(host):
        raise AssertionError("test attempted forbidden DNS: " + repr(host))
    return original_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = connect
socket.getaddrinfo = getaddrinfo
