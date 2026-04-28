# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import argparse
import math

import torch
import pandas as pd

# Requires flash-linear-attention==0.5.0
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_fla,
)
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_fla,
)

from flash_qla import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_qla
from flash_qla import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_qla
from flash_qla.utils import l2norm, pack, profile

from ref_gdr import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_ref
from ref_gdr import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_ref


def test_gated_delta_rule(
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    varlen: bool = False,
    cu_seqlens: list[int] | None = None,
    use_h0: bool = False,
    chunk_size: int = 64,
    data_dtype: str = "bfloat16",
    ref_dtype: str = "float32",
    device: torch.device = "cuda",
    random_seed: int = 42,
    check_accuracy: bool = True,
    show_speedup: bool = True,
    auto_cp: bool = True,
    swa_ratio: float = 0.75,
    skip_bwd: bool = False,
):
    data_dtype = getattr(torch, data_dtype)
    ref_dtype = getattr(torch, ref_dtype)
    torch.manual_seed(random_seed)
    q = l2norm(
        torch.randn(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            device=device,
            dtype=data_dtype,
        )
    )
    k = l2norm(
        torch.randn(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            device=device,
            dtype=data_dtype,
        )
    )
    v = torch.randn(
        (batch_size, num_tokens, num_v_heads, head_dim_v),
        device=device,
        dtype=data_dtype,
    )
    g = (
        torch.nn.functional.logsigmoid(
            torch.randn(
                (batch_size, num_tokens, num_v_heads),
                device=device,
                dtype=torch.float32,
            )
        )
        / 16
    )
    beta = torch.randn(
        (batch_size, num_tokens, num_v_heads), device=device, dtype=torch.float32
    ).sigmoid()
    h0 = (
        torch.randn(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            device=device,
            dtype=torch.float32,
        )
        if use_h0
        else None
    )
    do = torch.randn_like(v)
    dht = (
        torch.randn(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            device=device,
            dtype=torch.float32,
        )
        / 8
        if use_h0
        else None
    )
    scale = head_dim_k ** (-0.5)
    print(
        f"Shape: B={batch_size} Hk={num_k_heads} Hv={num_v_heads} T={num_tokens} VarLen={varlen}"
    )

    swa_mask = torch.zeros((num_v_heads), dtype=torch.bool, device=device)
    swa_mask[: math.ceil(swa_ratio * num_v_heads)] = 1
    swa_mask = swa_mask[torch.randperm(num_v_heads, device=device)]
    g[:, :, ~swa_mask] = 0.0
    print(f"SWA Mask: {swa_mask.to(torch.int32, copy=True).tolist()}")

    if varlen:
        if cu_seqlens is None:
            cu_seqlens = torch.randint(
                1, num_tokens, (batch_size,), device=device, dtype=torch.int32
            )
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(cu_seqlens, dim=-1), (1, 0)
            )
            q = pack(q, cu_seqlens)
            k = pack(k, cu_seqlens)
            v = pack(v, cu_seqlens)
            g = pack(g, cu_seqlens)
            beta = pack(beta, cu_seqlens)
            do = pack(do, cu_seqlens)
        else:
            assert batch_size == 1
            assert cu_seqlens[0] == 0
            assert cu_seqlens[-1] == num_tokens
            cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
            if use_h0:
                real_batch_size = cu_seqlens.shape[0] - 1
                h0 = torch.randn(
                    (real_batch_size, num_v_heads, head_dim_k, head_dim_v),
                    device=device,
                    dtype=torch.float32,
                )
                dht = (
                    torch.randn(
                        (real_batch_size, num_v_heads, head_dim_k, head_dim_v),
                        device=device,
                        dtype=torch.float32,
                    )
                    / 8
                )
            assert (cu_seqlens[1:] - cu_seqlens[:-1]).min() > 0
    else:
        cu_seqlens = None

    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.to(ref_dtype, copy=True),
        k=k.to(ref_dtype, copy=True),
        v=v.to(ref_dtype, copy=True),
        g=g.to(ref_dtype, copy=True),
        beta=beta.to(ref_dtype, copy=True),
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
    )
    g_fla, o_fla, A_fla, s_fla, _, _ = chunk_gated_delta_rule_fwd_fla(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    g_qla, A_qla, o_qla, h_qla, s_qla = chunk_gated_delta_rule_fwd_qla(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        output_h=True,
        auto_cp=auto_cp,
    )

    if check_accuracy:
        print(
            f"h_qla: {(h_qla - h_ref).abs().max().item():.4f} / {h_ref.abs().max().item():.4f}"
        )
        print(
            f"s_fla: {(s_fla - s_ref).abs().max().item():.4f} / {s_ref.abs().max().item():.4f}"
        )
        print(
            f"s_qla: {(s_qla - s_ref).abs().max().item():.4f} / {s_ref.abs().max().item():.4f}"
        )
        print(
            f"o_fla: {(o_fla - o_ref).abs().max().item():.4f} / {o_ref.abs().max().item():.4f}"
        )
        print(
            f"o_qla: {(o_qla - o_ref).abs().max().item():.4f} / {o_ref.abs().max().item():.4f}"
        )

        for _ in range(1000):
            g_qla, A_qla, o_qla, h_qla, s_qla = chunk_gated_delta_rule_fwd_qla(
                q,
                k,
                v,
                g,
                beta,
                scale,
                h0,
                cu_seqlens,
                True,
                False,
                auto_cp,
            )
            try:
                if h0 is not None:
                    assert (
                        s_qla - s_ref
                    ).abs().max().item() <= s_ref.abs().max().item() * 0.02
                assert (
                    o_qla - o_ref
                ).abs().max().item() <= o_ref.abs().max().item() * 0.02
            except AssertionError as e:
                print("********** ERROR **********")
                if h0 is not None:
                    print(
                        f"s_qla: {(s_qla - s_ref).abs().max().item():.4f} / {s_ref.abs().max().item():.4f}"
                    )
                print(
                    f"o_qla: {(o_qla - o_ref).abs().max().item():.4f} / {o_ref.abs().max().item():.4f}"
                )
                print("********** ERROR **********")
                raise e

    if show_speedup:
        prof_fla = profile(
            chunk_gated_delta_rule_fwd_fla,
            [q, k, v, g, beta, scale, h0, True, cu_seqlens],
        )
        prof_qla = profile(
            chunk_gated_delta_rule_fwd_qla,
            [q, k, v, g, beta, scale, h0, cu_seqlens, True, False, auto_cp],
        )
        result_fla = {
            "[fwd] csum": prof_fla["chunk_local_cumsum_scalar_kernel"],
            "[fwd] solve": prof_fla["chunk_gated_delta_rule_fwd_kkt_solve_kernel"],
            "[fwd] wu": prof_fla["recompute_w_u_fwd_kernel"],
            "[fwd] gdr": prof_fla["chunk_gated_delta_rule_fwd_kernel_h_blockdim64"],
            "[fwd] o": prof_fla["chunk_fwd_kernel_o"],
        }
        result_qla = {
            "[fwd] csum": prof_qla["tilelang_chunk_local_cumsum_kernel_kernel"],
            "[fwd] solve": prof_qla["tilelang_kkt_solve_kernel_kernel"],
            "[fwd] gdr": prof_qla["tilelang_fused_chunk_gdr_fwd_kernel_kernel"],
        }
        if "tilelang_get_warmup_chunks_kernel_kernel" in prof_qla.keys():
            result_fla["[fwd] cp-w"] = None
            result_fla["[fwd] cp-h"] = None
            result_fla["[fwd] cp-c"] = None
            result_qla["[fwd] cp-w"] = prof_qla[
                "tilelang_get_warmup_chunks_kernel_kernel"
            ]
            result_qla["[fwd] cp-h"] = prof_qla["tilelang_prepare_h_kernel_kernel"]
            result_qla["[fwd] cp-c"] = prof_qla["tilelang_correct_h0_kernel_kernel"]
        result_fla["total"] = prof_fla["total"]
        result_qla["total"] = prof_qla["total"]
        results = {
            "fla": result_fla,
            "flash_qla": result_qla,
        }
        df = pd.DataFrame(results)
        print(df.round(3))
        speedup = results["fla"]["total"] / results["flash_qla"]["total"]
        print(f"Speed up: {speedup:.2f}x")

    if skip_bwd:
        return

    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(ref_dtype, copy=True),
        k.to(ref_dtype, copy=True),
        v.to(ref_dtype, copy=True),
        g_ref,
        beta.to(ref_dtype, copy=True),
        A_ref.to(ref_dtype, copy=True),
        scale,
        h0,
        do.to(ref_dtype, copy=True),
        dht,
        cu_seqlens,
    )
    dq_fla, dk_fla, dv_fla, db_fla, dg_fla, dh0_fla, _, _ = (
        chunk_gated_delta_rule_bwd_fla(
            q,
            k,
            v,
            g_fla,
            beta,
            A_fla,
            scale,
            h0,
            do,
            dht,
            cu_seqlens,
        )
    )
    dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = chunk_gated_delta_rule_bwd_qla(
        q,
        k,
        v,
        g_qla,
        beta,
        A_qla,
        do,
        dht,
        scale,
        h0,
        cu_seqlens,
    )

    if check_accuracy:
        print(
            f"dq_fla: {(dq_fla - dq_ref).abs().max().item():.4f} / {dq_ref.abs().max().item():.4f}"
        )
        print(
            f"dq_qla: {(dq_qla - dq_ref).abs().max().item():.4f} / {dq_ref.abs().max().item():.4f}"
        )
        print(
            f"dk_fla: {(dk_fla - dk_ref).abs().max().item():.4f} / {dk_ref.abs().max().item():.4f}"
        )
        print(
            f"dk_qla: {(dk_qla - dk_ref).abs().max().item():.4f} / {dk_ref.abs().max().item():.4f}"
        )
        print(
            f"dv_fla: {(dv_fla - dv_ref).abs().max().item():.4f} / {dv_ref.abs().max().item():.4f}"
        )
        print(
            f"dv_qla: {(dv_qla - dv_ref).abs().max().item():.4f} / {dv_ref.abs().max().item():.4f}"
        )
        if dht is not None:
            print(
                f"dh0_fla: {(dh0_fla - dh0_ref).abs().max().item():.4f} / {dh0_ref.abs().max().item():.4f}"
            )
            print(
                f"dh0_qla: {(dh0_qla - dh0_ref).abs().max().item():.4f} / {dh0_ref.abs().max().item():.4f}"
            )
        print(
            f"db_fla: {(db_fla - db_ref).abs().max().item():.4f} / {db_ref.abs().max().item():.4f}"
        )
        print(
            f"db_qla: {(db_qla - db_ref).abs().max().item():.4f} / {db_ref.abs().max().item():.4f}"
        )
        print(
            f"dg_fla: {(dg_fla - dg_ref).abs().max().item():.4f} / {dg_ref.abs().max().item():.4f}"
        )
        print(
            f"dg_qla: {(dg_qla - dg_ref).abs().max().item():.4f} / {dg_ref.abs().max().item():.4f}"
        )

        for _ in range(1000):
            dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = (
                chunk_gated_delta_rule_bwd_qla(
                    q,
                    k,
                    v,
                    g_qla,
                    beta,
                    A_qla,
                    do,
                    dht,
                    scale,
                    h0,
                    cu_seqlens,
                )
            )
            try:
                assert (
                    dq_qla - dq_ref
                ).abs().max().item() <= dq_ref.abs().max().item() * 0.02
                assert (
                    dk_qla - dk_ref
                ).abs().max().item() <= dk_ref.abs().max().item() * 0.02
                assert (
                    dv_qla - dv_ref
                ).abs().max().item() <= dv_ref.abs().max().item() * 0.02
                assert (
                    dg_qla - dg_ref
                ).abs().max().item() <= dg_ref.abs().max().item() * 0.02
                assert (
                    db_qla - db_ref
                ).abs().max().item() <= db_ref.abs().max().item() * 0.02
                if dht is not None:
                    assert (
                        dh0_qla - dh0_ref
                    ).abs().max().item() <= dh0_ref.abs().max().item() * 0.02
            except AssertionError as e:
                print("********** ERROR **********")
                print(
                    f"dq_qla: {(dq_qla - dq_ref).abs().max().item():.4f} / {dq_ref.abs().max().item():.4f}"
                )
                print(
                    f"dk_qla: {(dk_qla - dk_ref).abs().max().item():.4f} / {dk_ref.abs().max().item():.4f}"
                )
                print(
                    f"dv_qla: {(dv_qla - dv_ref).abs().max().item():.4f} / {dv_ref.abs().max().item():.4f}"
                )
                if dht is not None:
                    print(
                        f"dh0_qla: {(dh0_qla - dh0_ref).abs().max().item():.4f} / {dh0_ref.abs().max().item():.4f}"
                    )
                print(
                    f"db_qla: {(db_qla - db_ref).abs().max().item():.4f} / {db_ref.abs().max().item():.4f}"
                )
                print(
                    f"dg_qla: {(dg_qla - dg_ref).abs().max().item():.4f} / {dg_ref.abs().max().item():.4f}"
                )
                print("********** ERROR **********")
                raise e

    if show_speedup:
        prof_fla = profile(
            chunk_gated_delta_rule_bwd_fla,
            [q, k, v, g_fla, beta, A_fla, scale, h0, do, dht, cu_seqlens],
        )
        prof_qla = profile(
            chunk_gated_delta_rule_bwd_qla,
            [q, k, v, g_qla, beta, A_qla, do, dht, scale, h0, cu_seqlens],
        )
        result_fla = {
            "[bwd] csum": prof_fla["chunk_local_cumsum_scalar_kernel"],
            "[bwd] recom": prof_fla["recompute_w_u_fwd_kernel"]
            + prof_fla["chunk_gated_delta_rule_fwd_kernel_h_blockdim64"],
            "[bwd] dv": prof_fla["chunk_bwd_kernel_dv_local"],
            "[bwd] gdr": prof_fla["chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"],
            "[bwd] dqkwg": prof_fla["kernel_kernel"],
            "[bwd] wy": prof_fla["prepare_wy_repr_bwd_kernel"],
        }
        result_qla = {
            "[bwd] csum": prof_qla["tilelang_chunk_local_cumsum_kernel_kernel"],
            "[bwd] recom": prof_qla["tilelang_prepare_h_kernel_kernel"],
            "[bwd] gdr": prof_qla["tilelang_fused_chunk_gdr_bwd_kernel_kernel"],
        }
        if num_k_heads < num_v_heads:
            result_fla["[bwd] reduc"] = prof_fla["compress_heads_kernel"]
            result_qla["[bwd] reduc"] = (
                prof_qla["tilelang_group_reduce_vector_kernel_kernel"] * 2
            )
        result_fla["total"] = prof_fla["total"]
        result_qla["total"] = prof_qla["total"]
        results = {
            "fla": result_fla,
            "flash_qla": result_qla,
        }
        df = pd.DataFrame(results)
        print(df.round(3))
        speedup = results["fla"]["total"] / results["flash_qla"]["total"]
        print(f"Speed up: {speedup:2.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gated Delta Rule")
    parser.add_argument(
        "--set",
        type=str,
        default="develop",
        help="Preset name (loads from settings/{set}.csv)",
    )
    parser.add_argument(
        "--seqlen", "--num-tokens", type=int, default=16384, help="Sequence Length"
    )
    parser.add_argument(
        "--nkh",
        "--num-k-heads",
        type=int,
        default=0,
        help="Number of K heads (num_k_heads)",
    )
    parser.add_argument(
        "--nvh",
        "--num-heads",
        "--num-v-heads",
        type=int,
        default=64,
        help="Number of V heads (num_v_heads)",
    )
    parser.add_argument(
        "--no-h0",
        action="store_true",
        help="Disable initial state and gradient of final state",
    )
    parser.add_argument("--skip-bwd", action="store_true", help="Test forward only")
    parser.add_argument(
        "--no-cp",
        "--disable-auto-cp",
        action="store_true",
        help="Disable auto intra-card CP",
    )
    parser.add_argument(
        "--swa-ratio", type=float, default=0.75, help="Ratio of sliding-window heads"
    )
    parser.add_argument(
        "--data-dtype",
        type=str,
        default="bfloat16",
        help="Data type for input and output",
    )
    parser.add_argument(
        "--ref-dtype", type=str, default="float64", help="Data type for reference"
    )
    parser.add_argument("--hide-acc", action="store_true", help="Do not print accuracy")
    parser.add_argument("--hide-lat", action="store_true", help="Do not print latency")
    parser.add_argument(
        "--seed", "--random-seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    if args.nkh <= 0:
        args.nkh = args.nvh

    metadata = {
        "head_dim_k": 128,  # MUST BE 128
        "head_dim_v": 128,  # MUST BE 128
        "chunk_size": 64,  # MUST BE 64
        "num_tokens": args.seqlen,
        "num_k_heads": args.nkh,
        "num_v_heads": args.nvh,
        "use_h0": not args.no_h0,
        "data_dtype": args.data_dtype,
        "ref_dtype": args.ref_dtype,
        "check_accuracy": not args.hide_acc,
        "show_speedup": not args.hide_lat,
        "skip_bwd": args.skip_bwd,
        "auto_cp": not args.no_cp,
        "swa_ratio": args.swa_ratio,
        "random_seed": args.seed,
        "device": "cuda",
    }

    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    preset = pd.read_csv(os.path.join(script_dir, "settings", f"{args.set}.csv"))
    for i, row in preset.iterrows():
        print("-" * 64)
        torch.cuda.empty_cache()
        data = row.to_dict()
        if "cu_seqlens" in data.keys():
            data["cu_seqlens"] = list(map(int, data["cu_seqlens"].split("-")))
        metadata.update(data)
        test_gated_delta_rule(**metadata)
    print("-" * 64)
