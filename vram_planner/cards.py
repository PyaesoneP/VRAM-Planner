"""Stored model cards: everything the planner needs from a GGUF, without the GGUF.

analyze() touches the model file for exactly three things - extract_config(),
classify_tensors() and find_mmproj(). All three are pure functions of the header
and the tensor table, and together they are a few kilobytes. Everything after
them is arithmetic on those numbers.

So they are cached. A card is written every time a real file is read, which makes
the library accumulate for free, and analyze() falls back to it when the file is
gone. That buys three things:

  * plan for a model you have deleted, or have not downloaded yet
  * plan on a machine that does not hold the weights at all
  * re-derive stored calibration rows whose model file has since been removed,
    instead of stranding them as unreliable (see calib._recompute_overheads)

A card is NOT a substitute for the file when the file exists: a card is keyed by
file name, and if the name is the same but the size is not, the card is stale and
gets rebuilt. Two different models under one file name is the one thing this
cannot survive, and it is also not a thing that happens by accident.
"""
import json, os, time
from .paths import _user_file

CARD_SCHEMA = 1

# classify_tensors() keys these by layer index. JSON has no integer keys, so they
# come back as strings and every `per_layer[i]` lookup silently misses - which
# reads as a model with no layers rather than as an error.
_INT_KEYED = ("per_layer_bytes", "per_layer_expert_bytes", "per_layer_ffn_bytes")


def _cards_store():
    return _user_file("model_cards.json")


def load_cards():
    try:
        d = json.load(open(_cards_store(), encoding="utf-8"))
        return d if isinstance(d, dict) and "cards" in d else {"cards": {}}
    except Exception:
        return {"cards": {}}


def save_cards(data):
    try:
        json.dump(data, open(_cards_store(), "w", encoding="utf-8"), indent=1)
        return True
    except Exception:
        return False


def card_name(path):
    """Cards are keyed by file name, not full path: the same model moves between
    drives and folders, and a plan should not care where it used to live."""
    return os.path.basename(path or "")


def make_card(path, cfg, cl, mmproj, file_bytes=None, n_shards=1):
    """`file_bytes` and `n_shards` come from the parsed GGUF rather than from
    os.path.getsize, because a sharded model's size is the sum over its shards -
    which is what the planner reports as the on-disk figure."""
    card = {"schema": CARD_SCHEMA, "name": card_name(path), "when": int(time.time()),
            "config": dict(cfg), "tensors": dict(cl), "mmproj": None,
            "n_shards": int(n_shards or 1)}
    if file_bytes is None:
        try:
            file_bytes = os.path.getsize(path)
        except OSError:
            file_bytes = 0
    card["file_bytes"] = int(file_bytes or 0)
    for k in _INT_KEYED:
        if k in card["tensors"]:
            card["tensors"][k] = {str(i): v for i, v in card["tensors"][k].items()}
    # Sets do not serialise. extract_config already returns these sorted, but a
    # future field that does not would corrupt cards silently rather than raise.
    for k, v in list(card["config"].items()):
        if isinstance(v, set):
            card["config"][k] = sorted(v)
    if mmproj:
        # The absolute path is deliberately dropped - it is meaningless once the
        # file is gone, and everything the planner reads is these four fields.
        card["mmproj"] = {"name": mmproj.get("name"), "bytes": mmproj.get("bytes"),
                          "tensor_bytes": mmproj.get("tensor_bytes"),
                          "vision": mmproj.get("vision")}
    return card


def _rehydrate(card):
    """(config, classify, mmproj, meta) from a stored card, int keys restored."""
    cfg = dict(card.get("config") or {})
    cl = dict(card.get("tensors") or {})
    for k in _INT_KEYED:
        if k in cl:
            cl[k] = {int(i): v for i, v in (cl[k] or {}).items()}
    mm = card.get("mmproj")
    if mm:
        mm = dict(mm, path="")            # no file behind it; callers only read sizes
    meta = {"file_bytes": card.get("file_bytes") or 0,
            "n_shards": card.get("n_shards") or 1,
            "when": card.get("when")}
    return cfg, cl, mm, meta


def remember_card(path, cfg, cl, mmproj, file_bytes=None, n_shards=1):
    """Write-through cache. Skips the write when an identical card is already
    stored, because analyze() runs in tight loops (calibration, sweeps) and this
    must not turn every plan into a disk write."""
    name = card_name(path)
    if not name:
        return False
    data = load_cards()
    old = (data["cards"] or {}).get(name)
    size = file_bytes
    if size is None:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
    if old and old.get("file_bytes") == size and old.get("schema") == CARD_SCHEMA:
        return False
    data["cards"][name] = make_card(path, cfg, cl, mmproj, size, n_shards)
    return save_cards(data)


def load_card(path_or_name):
    """(config, classify, mmproj, meta), or None. Path or bare file name."""
    card = (load_cards()["cards"] or {}).get(card_name(path_or_name))
    if not card or card.get("schema") != CARD_SCHEMA:
        return None
    try:
        return _rehydrate(card)
    except Exception:
        return None


def have_card(path_or_name):
    c = (load_cards()["cards"] or {}).get(card_name(path_or_name))
    return bool(c and c.get("schema") == CARD_SCHEMA)


def forget_card(name):
    data = load_cards()
    if card_name(name) in (data["cards"] or {}):
        del data["cards"][card_name(name)]
        save_cards(data)
        return True
    return False


def list_cards():
    """[{name, when, file_bytes, arch, n_layers, params, has_mmproj, on_disk}]"""
    out = []
    for name, c in sorted((load_cards()["cards"] or {}).items()):
        cfg, cl = c.get("config") or {}, c.get("tensors") or {}
        out.append({"name": name, "when": c.get("when"),
                    "file_bytes": c.get("file_bytes") or 0,
                    "arch": cfg.get("arch") or "?",
                    "n_layers": cfg.get("n_layers") or 0,
                    "params": cl.get("params_total") or 0,
                    "weights_bytes": cl.get("weights_total") or 0,
                    "has_mmproj": bool(c.get("mmproj"))})
    return out
