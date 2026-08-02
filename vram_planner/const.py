"""Version, byte units, and the MiB helper every other module uses."""


__version__ = "1.0.3"


MiB = 1024 * 1024

GiB = 1024 * 1024 * 1024


def _mib(x):
    return x / MiB
