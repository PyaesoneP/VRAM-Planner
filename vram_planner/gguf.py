"""GGUF binary reader: header, metadata, tensor table, shard following."""
import os, re, struct


# ---------------------------------------------------------------------------
# GGML tensor type table: id -> (name, block_size_elements, bytes_per_block)
# tensor bytes = ceil(n_elements / block) * bytes_per_block
# ---------------------------------------------------------------------------
GGML_TYPES = {
    0:  ("F32",      1,   4),
    1:  ("F16",      1,   2),
    2:  ("Q4_0",    32,  18),
    3:  ("Q4_1",    32,  20),
    6:  ("Q5_0",    32,  22),
    7:  ("Q5_1",    32,  24),
    8:  ("Q8_0",    32,  34),
    9:  ("Q8_1",    32,  40),
    10: ("Q2_K",   256,  84),
    11: ("Q3_K",   256, 110),
    12: ("Q4_K",   256, 144),
    13: ("Q5_K",   256, 176),
    14: ("Q6_K",   256, 210),
    15: ("Q8_K",   256, 292),
    16: ("IQ2_XXS",256,  66),
    17: ("IQ2_XS", 256,  74),
    18: ("IQ3_XXS",256,  98),
    19: ("IQ1_S",  256,  50),
    20: ("IQ4_NL",  32,  18),
    21: ("IQ3_S",  256, 110),
    22: ("IQ2_S",  256,  82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8",       1,   1),
    25: ("I16",      1,   2),
    26: ("I32",      1,   4),
    27: ("I64",      1,   8),
    28: ("F64",      1,   8),
    29: ("IQ1_M",  256,  56),
    30: ("BF16",     1,   2),
    31: ("TQ1_0",  256,  54),
    32: ("TQ2_0",  256,  66),
    39: ("MXFP4",   32,  17),   # gpt-oss; best-effort, validated against file size
}


# GGUF metadata value types
_SCALAR_FMT = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}

_T_STRING, _T_ARRAY = 8, 9


def _r(f, fmt):
    size = struct.calcsize('<' + fmt)
    data = f.read(size)
    if len(data) < size:
        raise EOFError("unexpected end of file while reading header")
    return struct.unpack('<' + fmt, data)[0]


def _read_str(f):
    n = _r(f, 'Q')
    return f.read(n).decode('utf-8', errors='replace')


def _read_value(f, vtype):
    """Read one metadata value. Big arrays are skipped (only their bytes consumed)."""
    if vtype in _SCALAR_FMT:
        return _r(f, _SCALAR_FMT[vtype])
    if vtype == _T_STRING:
        return _read_str(f)
    if vtype == _T_ARRAY:
        elem_type = _r(f, 'I')
        count = _r(f, 'Q')
        if elem_type == _T_STRING:
            for _ in range(count):
                ln = _r(f, 'Q'); f.seek(ln, 1)
            return None
        if elem_type == _T_ARRAY:
            for _ in range(count):
                _read_value(f, _T_ARRAY)
            return None
        fmt = _SCALAR_FMT.get(elem_type)
        if fmt is None:
            return None
        esize = struct.calcsize('<' + fmt)
        if count <= 8192:                      # keep small numeric arrays (e.g. per-layer kv heads)
            raw = f.read(esize * count)
            return list(struct.unpack('<' + fmt * count, raw))
        f.seek(esize * count, 1)               # skip huge arrays (tokenizer vocab, etc.)
        return None
    raise ValueError("unknown GGUF value type %d" % vtype)


SHARD_RE = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)


def _find_shards(path):
    """If path is one shard of a split GGUF, return all sibling shards in order."""
    d = os.path.dirname(path)
    base = os.path.basename(path)
    m = SHARD_RE.match(base)
    if not m:
        return [path]
    prefix, _, total = m.group(1), m.group(2), int(m.group(3))
    shards = []
    for i in range(1, total + 1):
        cand = os.path.join(d, "%s-%05d-of-%05d.gguf" % (prefix, i, total))
        if os.path.exists(cand):
            shards.append(cand)
    return shards or [path]


def _parse_one(path):
    """Parse a single GGUF file: returns (version, metadata_dict, tensor_list)."""
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError("not a GGUF file (bad magic): %s" % path)
        version = _r(f, 'I')
        tensor_count = _r(f, 'Q')
        kv_count = _r(f, 'Q')
        meta = {}
        for _ in range(kv_count):
            key = _read_str(f)
            vtype = _r(f, 'I')
            meta[key] = _read_value(f, vtype)
        tensors = []
        for _ in range(tensor_count):
            name = _read_str(f)
            n_dims = _r(f, 'I')
            dims = [_r(f, 'Q') for _ in range(n_dims)]
            type_id = _r(f, 'I')
            offset = _r(f, 'Q')
            n_elem = 1
            for d in dims:
                n_elem *= d
            tname, block, tsize = GGML_TYPES.get(type_id, ("TYPE_%d" % type_id, None, None))
            if block:
                nbytes = (n_elem // block) * tsize
                if n_elem % block:
                    nbytes += tsize
                unknown = False
            else:
                nbytes = 0
                unknown = True
            tensors.append({
                "name": name, "dims": dims, "type_id": type_id, "type_name": tname,
                "n_elements": n_elem, "n_bytes": nbytes, "unknown_type": unknown,
                "offset": offset,
            })
        # The data section starts at the first alignment boundary after the tensor
        # table. Needed to size the last tensor from the file length.
        align = _as_int_meta(meta.get("general.alignment"), 32) or 32
        pos = f.tell()
        data_start = -(-pos // align) * align
    _size_unknown_from_offsets(tensors, data_start, os.path.getsize(path))
    return version, meta, tensors


def _as_int_meta(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _size_unknown_from_offsets(tensors, data_start, file_size):
    """Size tensors of an unrecognised quant from the gaps between tensor offsets.

    A new GGML type used to count as zero bytes, which silently understated the
    weights - the one number this tool advertises as exact. But the type table is
    not the only source: tensors are laid out back to back in the data section, so
    the space a tensor occupies is simply the distance to the next one, and the
    last one runs to the end of the file. That is exact and needs no type table at
    all, which is the point - it holds for quants that do not exist yet.

    Only unknown types are filled in this way. Known ones keep their computed size,
    because the gap includes any inter-tensor padding and would overstate them."""
    if not tensors or not any(t["unknown_type"] for t in tensors):
        return
    order = sorted(tensors, key=lambda t: t["offset"])
    data_bytes = max(0, file_size - data_start)
    for i, t in enumerate(order):
        if not t["unknown_type"]:
            continue
        end = order[i + 1]["offset"] if i + 1 < len(order) else data_bytes
        gap = end - t["offset"]
        if gap > 0:
            t["n_bytes"] = gap
            t["sized_from_offsets"] = True


def load_gguf(path):
    """Load a model (following shards). Metadata comes from the first shard."""
    shards = _find_shards(path)
    version, meta, tensors = _parse_one(shards[0])
    file_bytes = os.path.getsize(shards[0])
    for s in shards[1:]:
        _, _, extra = _parse_one(s)
        tensors.extend(extra)
        file_bytes += os.path.getsize(s)
    return {"path": path, "shards": shards, "version": version,
            "meta": meta, "tensors": tensors, "file_bytes": file_bytes}


def parse_meta_only(path):
    """Fast: read header + metadata only (skip the tensor table). Used during scan."""
    with open(path, "rb") as f:
        if f.read(4) != b'GGUF':
            raise ValueError("bad magic")
        _r(f, 'I')                       # version
        _r(f, 'Q')                       # tensor_count (ignored)
        kv_count = _r(f, 'Q')
        meta = {}
        for _ in range(kv_count):
            key = _read_str(f)
            meta[key] = _read_value(f, _r(f, 'I'))
    return meta
