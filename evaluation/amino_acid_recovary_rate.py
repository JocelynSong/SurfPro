import numpy as np
import os

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment
import json


def compute_aa_recovery_rate(generation_file, target_file):
    gens = open(generation_file, "r", encoding="utf-8").readlines()
    lines = open(target_file, "r", encoding="utf-8").readlines()

    rates = []
    for idx, gen in enumerate(gens):
        line = lines[idx]

        seq1 = Seq(line.strip())
        seq2 = Seq(gen.strip())

        # Finding similarities
        alignments = pairwise2.align.globalxx(seq1, seq2)
        for alignment in alignments[: 1]:
            print(alignment.seqA)
            print(alignment.seqB)
            terms = format_alignment(*alignment).split()
            score = int(terms[-1].replace("Score=", ""))
            num = len(terms[-2])
            rate = float(score) / num
            print(rate)
            rates.append(rate)
    print(len(rates))
    print(np.mean(np.array(rates)))


if __name__ == "__main__":
    data_path = "cath42"
    target_path = os.path.join(data_path, "src.seq.txt")
    output_path = os.path.join(data_path, "protein.txt")
    compute_aa_recovery_rate(output_path, target_path)
