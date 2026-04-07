"""
ECE548 - Grad Project
Reed-Solomon Error Correction for PUF Key Generation

Compares RS performance under two error models:
    1. Independent bit errors (worst case for RS)
    2. Correlated / clustered bit errors (best case for RS)

Uses RS(255, k) over GF(2^8) where k = 255 - nsym.
Total codeword is always 255 symbols (the natural block length).

Requirements: numpy, matplotlib, reedsolo

Usage:
    pip install reedsolo numpy matplotlib
    python puf_rs_simulation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from reedsolo import RSCodec, ReedSolomonError
from collections import defaultdict


# Configuration

SEED = 42
NUM_TRIALS = 100
CODEWORD_LEN = 255  # RS block length over GF(2^8)
SYMBOL_BITS = 8
BER_RANGE = np.arange(0.01, 0.21, 0.02)

RS_CONFIGS = [
    {"nsym": 20, "label": "RS(255,235) t=10"},
    {"nsym": 50, "label": "RS(255,205) t=25"},
    {"nsym": 80, "label": "RS(255,175) t=40"},
    {"nsym": 120, "label": "RS(255,135) t=60"},
]

CLUSTER_SIZE_MEAN = 6   # avg cluster width in bits
CLUSTER_SIZE_STD = 2


# PUF Simulation

# Generates the enrollment reading. Creates k random bytes (each between
# 0 and 255), where each byte is one symbol. This represents the clean
# first reading of the PUF that everything else is compared against.
def generate_puf_response(k, rng):
    """Random reference PUF response (k data symbols)."""
    return rng.integers(0, 256, size=k, dtype=np.uint8)


# Simulates a noisy second reading where errors are spread out. Unpacks
# every byte into its 8 individual bits, then for each bit rolls a random
# number. If that number is less than the BER, the bit gets flipped.
# Packs the bits back into bytes. The result is a copy of the original
# response where roughly BER percent of bits have been flipped randomly.
def apply_independent_errors(response, ber, rng):
    """Flip each bit independently with probability = ber."""
    bits = np.unpackbits(response)
    flip_mask = rng.random(len(bits)) < ber
    noisy_bits = np.bitwise_xor(bits, flip_mask.astype(np.uint8))
    return np.packbits(noisy_bits)[:len(response)]


# Simulates a noisy second reading where errors clump together. Calculates
# how many total bits should flip (total bits * BER) to match the same
# overall error rate as the independent model. Picks random center points
# and creates clusters around them. Each cluster width is drawn from a
# normal distribution. Keeps placing clusters until it hits the target
# number of flipped bits. Same total errors as independent, but bunched
# together in groups.
def apply_correlated_errors(response, ber, rng):
    """Clustered burst errors: same overall BER, but spatially grouped."""
    total_bits = len(response) * SYMBOL_BITS
    target_flips = int(total_bits * ber)
    if target_flips == 0:
        return response.copy()

    flip_positions = set()
    while len(flip_positions) < target_flips:
        center = rng.integers(0, total_bits)
        width = max(1, int(rng.normal(CLUSTER_SIZE_MEAN, CLUSTER_SIZE_STD)))
        start = max(0, center - width // 2)
        end = min(total_bits, start + width)
        for pos in range(start, end):
            flip_positions.add(pos)
            if len(flip_positions) >= target_flips:
                break

    bits = np.unpackbits(response)
    for pos in flip_positions:
        bits[pos] ^= 1
    return np.packbits(bits)[:len(response)]


# Compares two byte arrays symbol by symbol. Any position where the bytes
# differ counts as one symbol error, regardless of how many bits differ
# within that byte. This is the count that RS cares about.
def count_symbol_errors(original, noisy):
    return int(np.sum(original != noisy))


# Fuzzy Extractor

# One-time enrollment. Takes the clean PUF response and RS-encodes it,
# producing a 255-symbol codeword (the original k data symbols plus
# parity symbols). The entire codeword is returned as helper data that
# gets stored for later use.
def enrollment(reference, rs_codec):
    """Encode reference -> full codeword (255 symbols) as helper data."""
    encoded = rs_codec.encode(bytes(reference))
    return list(encoded)


# Reconstruction at every power-up. Takes the noisy second reading and
# the stored helper data. Pulls out just the parity portion from the
# helper data (everything after the first k symbols). Stitches together
# the noisy reading plus the original parity into a 255-symbol received
# word and hands it to the RS decoder. If the number of corrupted symbols
# is within the correction capacity, RS fixes them and returns True.
# If there are too many errors, RS throws an exception and returns False.
def reconstruction(noisy_response, helper_codeword, rs_codec):
    """
    Code-offset reconstruction:
    Replace data portion of stored codeword with noisy reading,
    keep original parity intact, then RS-decode.
    """
    k = len(noisy_response)
    parity = helper_codeword[k:]
    received = bytes(list(noisy_response) + parity)
    try:
        decoded = rs_codec.decode(received)
        # reedsolo returns (data, full_codeword, errata_positions)
        return True, bytes(decoded[0])
    except ReedSolomonError:
        return False, None


# Experiment

# Main loop. For each RS configuration and each BER value, runs NUM_TRIALS
# trials. Each trial generates a fresh PUF response, enrolls it, applies
# both noise models, attempts reconstruction under each, and records
# whether it succeeded and how many symbol errors there were. After all
# trials for a given BER, computes the failure rate (failures / total
# trials) and the average symbol error count.
def run_experiment():
    rng = np.random.default_rng(SEED)

    results = {"independent": defaultdict(list), "correlated": defaultdict(list)}
    sym_errs = {"independent": defaultdict(list), "correlated": defaultdict(list)}

    for rs_cfg in RS_CONFIGS:
        nsym = rs_cfg["nsym"]
        label = rs_cfg["label"]
        k = CODEWORD_LEN - nsym
        t = nsym // 2
        rs = RSCodec(nsym)

        print(f"\n {label} (k={k}, t={t}, PUF={k*8} bits)")

        for ber in BER_RANGE:
            fail_ind = fail_cor = 0
            se_ind_list, se_cor_list = [], []

            for _ in range(NUM_TRIALS):
                ref = generate_puf_response(k, rng)
                helper = enrollment(ref, rs)

                # Independent errors
                noisy_ind = apply_independent_errors(ref, ber, rng)
                se_ind_list.append(count_symbol_errors(ref, noisy_ind))
                ok, _ = reconstruction(noisy_ind, helper, rs)
                if not ok:
                    fail_ind += 1

                # Correlated errors
                noisy_cor = apply_correlated_errors(ref, ber, rng)
                se_cor_list.append(count_symbol_errors(ref, noisy_cor))
                ok, _ = reconstruction(noisy_cor, helper, rs)
                if not ok:
                    fail_cor += 1

            fr_ind = fail_ind / NUM_TRIALS
            fr_cor = fail_cor / NUM_TRIALS
            results["independent"][label].append(fr_ind)
            results["correlated"][label].append(fr_cor)
            sym_errs["independent"][label].append(np.mean(se_ind_list))
            sym_errs["correlated"][label].append(np.mean(se_cor_list))

            print(
                f"  BER={ber:.2f} | "
                f"Ind: fail={fr_ind:.3f} sym_err={np.mean(se_ind_list):5.1f}/{k} | "
                f"Cor: fail={fr_cor:.3f} sym_err={np.mean(se_cor_list):5.1f}/{k}"
            )

    return results, sym_errs


# Plotting

# Takes the results dictionaries and produces three figures from the
# recorded data. No computation happens here, purely visualization. 
def plot_results(results, sym_errs):
    ber_list = list(BER_RANGE)
    colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]

    # Figure 1: Failure rates stacked (independent on top, correlated below)
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    for i, cfg in enumerate(RS_CONFIGS):
        ax1.plot(ber_list, results["independent"][cfg["label"]],
                    marker="o", color=colors[i], label=cfg["label"], lw=2, ms=4)
    ax1.set(xlabel="Bit Error Rate (BER)", ylabel="Reconstruction Failure Rate",
            title="Independent Bit Errors", ylim=(-0.05, 1.05))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for i, cfg in enumerate(RS_CONFIGS):
        ax2.plot(ber_list, results["correlated"][cfg["label"]],
                    marker="s", color=colors[i], label=cfg["label"], lw=2, ms=4)
    ax2.set(xlabel="Bit Error Rate (BER)", ylabel="Reconstruction Failure Rate",
            title="Correlated / Clustered Bit Errors", ylim=(-0.05, 1.05))
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig("fig1_failure_rates.png", dpi=150, bbox_inches="tight")
    #fig1.savefig("fig1_failure_rates.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Symbol error rate vs correction capacity
    fig2, ax = plt.subplots(figsize=(7, 5))

    for i, cfg in enumerate(RS_CONFIGS):
        label = cfg["label"]
        t = cfg["nsym"] // 2
        ax.plot(ber_list, sym_errs["independent"][label],
                marker="o", color=colors[i], label="Indep", lw=2, ms=4)
        ax.plot(ber_list, sym_errs["correlated"][label],
                marker="s", ls="--", color=colors[i],
                label="Correl", lw=2, ms=4, alpha=0.7)
        ax.axhline(y=t, color=colors[i], ls=":", alpha=0.5, lw=1.5)
        ax.text(ber_list[-1] + 0.002, t, f"t={t}", color=colors[i],
                fontsize=7, va="center")
    custom = [
        Line2D([0], [0], color="gray", marker="o", lw=2, ms=4, label="Independent"),
        Line2D([0], [0], color="gray", marker="s", ls="--", lw=2, ms=4, label="Correlated"),
        Line2D([0], [0], color="gray", ls=":", lw=1.5, label="Correction limit t"),
    ]
    ax.legend(handles=custom, fontsize=8)
    ax.set(xlabel="Bit Error Rate (BER)", ylabel="Avg Symbol Errors",
            title="Symbol Error Rate vs Correction Capacity")
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("fig2_symbol_errors.png", dpi=150, bbox_inches="tight")
    #fig2.savefig("fig2_symbol_errors.pdf", bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: RS efficiency gain from correlated errors
    fig3, ax = plt.subplots(figsize=(7, 5))

    for i, cfg in enumerate(RS_CONFIGS):
        label = cfg["label"]
        ind = np.array(sym_errs["independent"][label])
        cor = np.array(sym_errs["correlated"][label])
        reduction = np.where(ind > 0, (ind - cor) / ind * 100, 0)
        ax.plot(ber_list, reduction, marker="D", color=colors[i],
                label=cfg["label"], lw=2, ms=4)
    ax.set(xlabel="Bit Error Rate (BER)", ylabel="Symbol Error Reduction (%)",
            title="RS Efficiency Gain from Correlated Errors")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig("fig3_efficiency_gain.png", dpi=150, bbox_inches="tight")
    #fig3.savefig("fig3_efficiency_gain.pdf", bbox_inches="tight")
    plt.close(fig3)

    print("\nPlots saved: fig1_failure_rates, fig2_symbol_errors, fig3_efficiency_gain (.png / .pdf)")


# Entry point

if __name__ == "__main__":
    print("=" * 70)
    print(" RS Error Correction for PUF Key Generation - Simulation")
    print("=" * 70)
    print(f" Trials per config : {NUM_TRIALS}")
    print(f" RS block length : {CODEWORD_LEN} symbols over GF(2^8)")
    print(f" BER range : {BER_RANGE[0]:.2f} - {BER_RANGE[-1]:.2f}")
    print(f" Cluster size : mean={CLUSTER_SIZE_MEAN}, std={CLUSTER_SIZE_STD} bits")
    print("=" * 70)

    results, sym_errs = run_experiment()
    plot_results(results, sym_errs)
    print("\nDone.")
