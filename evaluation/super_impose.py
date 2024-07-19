import os.path
import pickle
import esm
import torch
from prody import *
import biotite.structure as struc
from biotite.structure import AtomArray, Atom
from biotite.structure.io import save_structure
from sidechainnet.utils.measure import get_seq_coords_and_angles
import numpy as np
from numpy import nan
import argparse
import sys
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--protein", type=str, default="")
# FLAGS, _ = parser.parse_known_args()
# protein = FLAGS.protein

RESTYPE_1to3 = {
     "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN","E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
RES_ATOM14 = [
    [''] * 14,
    ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
]


def print_pdb(coord, seq, chain, indices=None):
    array = []
    for i in range(coord.shape[0]):
        idx = indices[i] + 1 if indices else i + 1
        aaname = seq[i]
        aid = ALPHABET.index(aaname)
        aaname = RESTYPE_1to3[aaname]
        for j,atom in enumerate(RES_ATOM14[aid]):
            if atom != '' and (coord[i, j] ** 2).sum() > 1e-4:
                atom = Atom(coord[i, j], chain_id=chain, res_id=idx, atom_name=atom, res_name=aaname, element=atom[0])
                array.append(atom)
    return array


def kabsch(A, B):
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.bmm(A_c.transpose(1,2), B_c)  # [B, 3, 3]
    U, S, V = torch.svd(H)
    # Flip
    sign = (torch.det(U) * torch.det(V) < 0.0)
    if sign.any():
        S[sign] = S[sign] * (-1)
        U[sign,:] = U[sign,:] * (-1)
    # Rotation matrix
    R = torch.bmm(V, U.transpose(1,2))  # [B, 3, 3]
    # Translation vector
    t = b_mean - torch.bmm(R, a_mean.transpose(1,2)).transpose(1,2)
    A_aligned = torch.bmm(R, A.transpose(1,2)).transpose(1,2) + t
    return A_aligned, R, t

# X: [B, N, 4, 3], R: [B, 3, 3], t: [B, 3]
def rigid_transform(X, R, t):
    B, N, L = X.size(0), X.size(1), X.size(2)
    X = X.reshape(B, N * L, 3)
    X = torch.bmm(R, X.transpose(1,2)).transpose(1,2) + t
    return X.view(B, N, L, 3)


def get_gen_protein_mapping(gen_file, src_file):
    proteins = open(gen_file).readlines()
    srcs = open(src_file,).readlines()
    gen_dict = {}
    for i in range(len(srcs)):
        gens = proteins[i*10: (i+1)*10]
        gen_dict[srcs[i].strip()] = []
        for idx, gen in enumerate(gens):
            gen_dict[srcs[i].strip()].append({"index": i*10+idx, "protein": gen.strip()})
    return gen_dict


def get_greedy_gen_dict(gen_file, src_file):
    proteins = open(gen_file).readlines()
    srcs = open(src_file, ).readlines()
    gen_dict = {}
    for i in range(len(srcs)):
        gens = proteins[i * 1: (i + 1) * 1]
        gen_dict[srcs[i].strip()] = []
        for idx, gen in enumerate(gens):
            gen_dict[srcs[i].strip()].append({"index": i * 1 + idx, "protein": gen.strip()})
    return gen_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=str, default="EGFR")
    FLAGS, _ = parser.parse_known_args()
    protein_test = FLAGS.protein

    data_path = "binder_design/Binder_Design_Data"
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    protein_list = ["H3", "IL7Ra", "InsulinR", "PDGFR", "TGFb", "TrkA"]
    test_dict = {"H3": 38, "IL7Ra": 7, "InsulinR": 31, "PDGFR": 28, "TGFb": 14, "TrkA": 4}

    output_all_path = "binder_output"
    src_file = os.path.join(output_all_path, "src.seq.txt")
    gen_file = os.path.join(output_all_path, "protein.txt")

    output_path = os.path.join(output_all_path, "superimpose")
    os.system("mkdir -p {}".format(output_path))

    binder_lines = open(src_file).readlines()
    gen_dict = get_greedy_gen_dict(gen_file, src_file)
    # gen_dict = get_gen_protein_mapping(gen_file, src_file)

    protein_index = 0
    for protein in protein_list:
        print(protein)
        binders_this = binder_lines[protein_index: protein_index + test_dict[protein]]

        protein_dict = {}
        with open("{}/{}.pkl".format(data_path, protein), 'rb') as f:
            test_data = pickle.load(f)

            # The pickle file is a list of tuples ((target_name, None, bind_label), binder_seq, binder_coords, target_seq, target_coords)
            for idx, item in enumerate(test_data):
                label = item[0][2]
                binder = item[1]
                if label == 1:
                    protein_dict[binder] = item

        for idx, binder in enumerate(binders_this):

            labels, aseq, acoords, bseq, bcoords = protein_dict[binder.strip()]

            this_gens = gen_dict[binder.strip()]

            for this_gen in this_gens:
                file_name = this_gen["index"]
                seq = this_gen["protein"]

                achain = print_pdb(acoords, aseq, chain='A')
                bchain = print_pdb(bcoords, bseq, chain='B')
                array = struc.array(achain + bchain)
                save_structure('{}/reference.pdb'.format(output_path), array)

                with torch.no_grad():
                    output = model.infer_pdb(seq)
                with open('{}/fold.pdb'.format(output_path), "w") as f:
                    f.write(output)
                binder_path = '{}/fold.pdb'.format(output_path)

                # superimpose
                try:
                    new_chain = parsePDB(binder_path, model=1)
                except:
                    print(binder_path)
                    continue
                _, coords, seq, _, _ = get_seq_coords_and_angles(new_chain)
                coords = coords.reshape((-1, 14, 3))

                length = min(len(seq), len(aseq))
                aseq = aseq[: length]
                acoords = acoords[: length]
                seq = seq[: length]
                coords = coords[: length]

                acoords = torch.tensor(acoords)
                coords = torch.tensor(coords)
                _, R, t = kabsch(coords[None,:,1], acoords[None,:,1])
                coords = rigid_transform(coords.unsqueeze(0), R, t).squeeze(0)

                achain = print_pdb(coords.numpy(), seq, chain='A')
                bchain = print_pdb(bcoords, bseq, chain='B')
                array = struc.array(achain + bchain)
                save_structure('{}/{}.pdb'.format(output_path, file_name), array)

                # clean up
                lines = []
                with open('{}/{}.pdb'.format(output_path, file_name)) as f:
                    for line in f:
                        if line.split()[4] == 'A':
                            lines.append(line.strip("\r\n"))

                lines.append('TER')

                with open('{}/{}.pdb'.format(output_path, file_name)) as f:
                    for line in f:
                        if line.split()[4] == 'B':
                            lines.append(line.strip("\r\n"))

                with open('{}/{}.pdb'.format(output_path, file_name), "w") as f:
                    for line in lines:
                        print(line, file=f)

        protein_index += test_dict[protein]