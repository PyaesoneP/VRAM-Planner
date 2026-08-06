"""Memory-bandwidth roofline for tokens/second."""
from .const import _mib
from .kv import kv_bytes_per_token_layer


GPU_EFF   = 0.85       # GPU streams weights contiguously; close to peak

RAM_EFF_HI = 0.65      # contiguous-ish CPU reads

RAM_EFF_LO = 0.25      # scattered expert gather + CPU matmul limits


def per_token_bytes(cfg, cl, gpu_blocks, ctx_fill, kv_type, cpu_head=True,
                    n_cpu_moe=0, n_cpu_ffn=0):
    """Bytes read per generated token, split by where they live.

    n_cpu_moe mirrors llama.cpp: the routed experts of the first N blocks sit in
    system RAM even though the rest of those blocks is on the GPU, so a block can
    contribute to both sides. n_cpu_ffn does the same for dense FFN weights, which
    is what the KV-on-GPU mode (-ot ffn=CPU) does. Unlike experts, a dense FFN runs
    on EVERY token, so those bytes are not discounted."""
    per = cl.get("per_layer_bytes") or {}
    exps = cl.get("per_layer_expert_bytes") or {}
    n_exp, n_used = cfg["n_expert"], cfg["n_expert_used"]
    frac = (n_used / n_exp) if (n_exp and n_used) else 1.0
    ffns = cl.get("per_layer_ffn_bytes") or {}
    on = set(gpu_blocks)
    # MTP blocks are resident but do not run during ordinary decoding, so their
    # weights are never streamed per token. Counting them would understate tok/s.
    mtp = set(cfg.get("mtp_layers") or [])
    gpu = cpu = 0.0
    for i in range(cfg["n_layers"] or 0):
        if i in mtp:
            continue
        dense = per.get(i, 0) - exps.get(i, 0)
        act_exp = exps.get(i, 0) * frac          # only n_used experts fire
        if i in on:
            if i < n_cpu_ffn and ffns.get(i):    # dense FFN pinned to CPU by -ot
                gpu += dense - ffns[i]
                cpu += ffns[i]
            else:
                gpu += dense
            if i < n_cpu_moe:
                cpu += act_exp                   # experts pinned to CPU by --n-cpu-moe
            else:
                gpu += act_exp
        else:
            cpu += dense + act_exp
    # token_embd is a row gather, not a full read; the output head is a real matmul
    head = cl.get("output_bytes", 0)
    if cpu_head:
        cpu += head
    else:
        gpu += head
    # KV grows as the context fills; read once per token on its own device. A
    # sliding-window layer only ever reads its window, so its per-token cost stops
    # growing there - that is the difference between a few hundred MiB and tens of
    # GiB per token on a long-context Gemma-class model.
    lens = cfg.get("kv_ctx_per_layer") or []
    fill = max(0, ctx_fill)
    for i in range(cfg["n_layers"] or 0):
        cap = lens[i] if i < len(lens) else fill
        b = kv_bytes_per_token_layer(cfg, kv_type, i) * min(fill, cap)
        if i in on:
            gpu += b
        else:
            cpu += b
    return {"gpu_bytes": gpu, "cpu_bytes": cpu,
            "gpu_mib": _mib(gpu), "cpu_mib": _mib(cpu), "expert_frac": frac}


def estimate_speed(cfg, cl, gpu_blocks, ctx_fill, kv_type,
                   bw_vram_gbs, bw_ram_gbs, cpu_head=True, ram_eff=None,
                   n_cpu_moe=0, n_cpu_ffn=0):
    """tok/s for generation. With ram_eff set (from a calibration) this returns a
    single number; without it, a hi/lo bracket."""
    b = per_token_bytes(cfg, cl, gpu_blocks, ctx_fill, kv_type, cpu_head,
                        n_cpu_moe, n_cpu_ffn)
    def tps(re_):
        t = b["gpu_bytes"] / (bw_vram_gbs * GPU_EFF * 1e9)
        if b["cpu_bytes"] > 0:
            t += b["cpu_bytes"] / (bw_ram_gbs * re_ * 1e9)
        return (1.0 / t) if t > 0 else 0.0
    out = dict(b)
    out["bw_vram_gbs"] = bw_vram_gbs
    out["bw_ram_gbs"] = bw_ram_gbs
    out["ctx_fill"] = ctx_fill
    if ram_eff:
        out["tok_s"] = tps(ram_eff)
        out["ram_eff"] = ram_eff
        out["calibrated"] = True
    else:
        out["tok_s_hi"] = tps(RAM_EFF_HI)
        out["tok_s_lo"] = tps(RAM_EFF_LO)
        out["calibrated"] = False
    return out


def solve_ram_eff(cfg, cl, gpu_blocks, ctx_fill, kv_type,
                  bw_vram_gbs, bw_ram_gbs, measured_tok_s, cpu_head=True,
                  n_cpu_moe=0):
    """Back out the effective RAM bandwidth fraction from one measured tok/s.
    The GPU side is assumed to run at GPU_EFF (contiguous streaming), so the
    whole residual lands on the CPU term - which is the term that actually
    varies. Returns None if the measurement leaves no room for a CPU term."""
    if not measured_tok_s or measured_tok_s <= 0:
        return None
    b = per_token_bytes(cfg, cl, gpu_blocks, ctx_fill, kv_type, cpu_head, n_cpu_moe)
    t_total = 1.0 / measured_tok_s
    t_gpu = b["gpu_bytes"] / (bw_vram_gbs * GPU_EFF * 1e9)
    t_cpu = t_total - t_gpu
    if b["cpu_bytes"] <= 0 or t_cpu <= 0:
        return None
    return b["cpu_bytes"] / (t_cpu * bw_ram_gbs * 1e9)
