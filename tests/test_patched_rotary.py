import torch
import rotary_emb
from vllm import pos_encoding_ops

def apply_rotary_eager(query, key, cos, sin):
    def _apply_rot(x, cos, sin):
        rotary_dim = cos.shape[-1]

        dtype = x.dtype
        x_upcast = x.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        x1 = x_upcast[..., :rotary_dim]
        x2 = x_upcast[..., rotary_dim : 2 * rotary_dim]

        # Flash Attention rotary_emb kernel casts everything to float, not sure why, so we do so here as well.
        x[..., :rotary_dim] = (x1 * cos - x2 * sin).to(dtype)
        x[..., rotary_dim : 2 * rotary_dim] = (x1 * sin + x2 * cos).to(dtype)

    _apply_rot(query, cos, sin)
    _apply_rot(key, cos, sin)

def apply_rotary_flash(query, key, cos, sin):
    def _apply_rot(x, cos, sin):
        rotary_dim = cos.shape[-1]
        x1 = x[..., :rotary_dim]
        x2 = x[..., rotary_dim : 2 * rotary_dim]

        rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)

    _apply_rot(query, cos, sin)
    _apply_rot(key, cos, sin)

def apply_rotary_vllm(query, key, cos, sin):
    head_size = query.shape[-1]

    #print("query", query.dtype)
    #print("key", key.dtype)
    #print("cos", cos.dtype)
    #print("sin", sin.dtype)

    # Inplace operation, updating query and key.
    pos_encoding_ops.rotary_embedding(
        query,
        key,
        head_size,
        cos,
        sin,
        True
    )

cos = torch.rand(1, 1, 64).to("cuda").to(torch.float16)
sin = torch.rand(1, 1, 64).to("cuda").to(torch.float16)

head_dim = 128
num_heads = 32

query_eager = torch.rand(1, num_heads, head_dim).to(torch.float16).to("cuda")
key_eager = torch.rand(1, num_heads, head_dim).to(torch.float16).to("cuda")

query_vllm = query_eager.clone()
query_flash = query_eager.clone()

key_vllm = key_eager.clone()
key_flash = key_eager.clone()

apply_rotary_eager(query_eager, key_eager, cos, sin)
apply_rotary_flash(query_flash, key_flash, cos, sin)
apply_rotary_vllm(query_vllm, key_vllm, cos.float(), sin.float())

def check_diff(a, b, a_name, b_name):
    print(f"Allclose {a_name}, {b_name}: {torch.allclose(a, b)}; Abs reldiff: {((a - b).abs() / (a.abs() + 1e-12)).mean()}")

check_diff(query_eager, query_vllm, "query_eager", "query_vllm")
check_diff(query_eager, query_flash, "query_eager", "query_flash")
check_diff(key_eager, key_vllm, "key_eager", "key_vllm")
check_diff(key_eager, key_flash, "key_eager", "key_flash")
