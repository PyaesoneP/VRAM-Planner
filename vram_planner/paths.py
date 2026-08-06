"""Where user data lives."""
import os


def _data_dir():
    """Where per-user state lives. Not next to the script: that breaks on a
    read-only or shared install, and it is how machine-identifying data ends up
    one `git add .` away from being published."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "vram-planner")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        return os.path.dirname(os.path.abspath(__file__))
    return d


def _user_file(name):
    """Path in the data dir, migrating a pre-1.0 copy from beside the script."""
    new = os.path.join(_data_dir(), name)
    old = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not os.path.exists(new) and os.path.exists(old):
        try:
            os.replace(old, new)
        except Exception:
            return old
    return new
