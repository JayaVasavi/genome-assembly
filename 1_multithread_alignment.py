# Stage 1 - classical pre-processing of long reads: near-duplicate removal.
#
# Long-read datasets contain many reads covering the same locus at near
# identity. Keeping all of them inflates the overlap graph without adding any
# layout information, so near-duplicates are removed before the graph is built.
#
#INPUT: Quality check completed fasta reads (for some of them when doenloaded itself they are quality-checked)
# OUTPUT: one retained read sequence per line, in input order. FASTA headers are
# not carried through, so downstream node indices refer to positions in this
# file; keep it alongside the input FASTA if provenance back to SRA records
# matters.
#
# Usage
# -----
#     python multithread_alignment.py \
#         --reads SRR27477754.fasta \
#         --out pairwise_alignment.txt
#
#     python multithread_alignment.py \
#         --reads reads.fasta --out output.txt --identity 0.999 --workers 8

import argparse
import sys
from Bio import pairwise2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Set from the command line in __main__; kept as module-level constants so the
# function signatures below are unchanged.
IDENTITY_THRESHOLD = 0.995
MAX_WORKERS = 15


def read_fasta_file(file_path):
    """Return the sequences of a FASTA file, in file order. Headers are discarded."""
    sequences = []
    with open(file_path, "r") as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences


def perform_alignment(seq_pair, removed_indices, lock):
    """Score one read pair; mark the higher-indexed read for removal if duplicate.

    Returns j when read j is removed, otherwise None.

    The early exit skips a pair as soon as either of its reads has already been
    removed by another worker: once a read is gone there is no point scoring it.
    This is what keeps the run tractable - the pair count is quadratic, and the
    skip rate rises as removals accumulate.
    """
    i, j, seq1, seq2 = seq_pair
    with lock:
        if j in removed_indices or i in removed_indices:
            return None
    S1, S2 = len(seq1), len(seq2)
    alignments = pairwise2.align.localms(seq1, seq2, 1, -1, -1, -1, score_only=True)
    # Progress indicator. Written from every worker, so on a terminal the lines
    # interleave; redirect stdout to a file and it grows large. Cosmetic only.
    sys.stdout.write(f"\rProcessing: {i}-{j}")
    sys.stdout.flush()
    min_value = min(S1, S2)
    if alignments > IDENTITY_THRESHOLD * min_value:
        with lock:
            removed_indices.add(j)
        return j
    return None


def main(a):
    """Remove near-duplicate reads from the list `a`; return the retained reads.

    All ordered pairs i < j are scored in a thread pool. `removed_indices` is
    shared across workers and guarded by `lock`, so a read removed by one worker
    is immediately visible to the others and their pairs are skipped.
    """
    removed_indices = set()
    lock = Lock()
    seq_pairs = [
        (i, j, a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a))
    ]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(perform_alignment, pair, removed_indices, lock): pair
            for pair in seq_pairs
        }
        for future in as_completed(futures):
            result = future.result()

    start_seq = [seq for idx, seq in enumerate(a) if idx not in removed_indices]
    return start_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove near-duplicate long reads before OLC graph construction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reads", required=True,
                        help="input FASTA of QC-passed, adapter-trimmed reads")
    parser.add_argument("--out", required=True,
                        help="output text file: one retained read per line")
    parser.add_argument("--identity", type=float, default=IDENTITY_THRESHOLD,
                        help="duplicate threshold, as a fraction of the shorter read")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="thread pool size")
    args = parser.parse_args()

    IDENTITY_THRESHOLD = args.identity
    MAX_WORKERS = args.workers

    sequences = read_fasta_file(args.reads)
    print(f"Total number of sequences: {len(sequences)}")

    a = sequences[:]
    cleaned_sequences = main(a)
    print(f"\nNumber of sequences after cleaning: {len(cleaned_sequences)}")
    print(f"Removed: {len(sequences) - len(cleaned_sequences)} "
          f"at >= {100 * IDENTITY_THRESHOLD:.1f}% identity")

    with open(args.out, "w") as file:
        for item in cleaned_sequences:
            file.write(f"{item}\n")
    print(f"Wrote {args.out}")
