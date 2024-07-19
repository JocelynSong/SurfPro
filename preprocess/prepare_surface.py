import os
import argparse
import random

import numpy as np
from random import sample
from sklearn.neighbors import NearestNeighbors
#from numba import jit
import time

amino_acid_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                   "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                   "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}



def extract_vertix(input_file):
    print(input_file)
    import codecs
    with codecs.open(input_file, 'r', encoding='utf-8',
                     errors='ignore') as fdata:
        lines = fdata.readlines()[3: ]
    vertices, atoms = [], []
    for line in lines:
        phrases = line.strip().split()
        aa = phrases[-1].split("_")[1]
        if aa not in amino_acid_dict:
            continue
        atoms.append(phrases[-1])
        vertices.append(np.array([float(phrases[0]), float(phrases[1]), float(phrases[2])]))

    if len(vertices) <= 1:
        return vertices, atoms

    nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(np.array(vertices))
    distances, indices = nbrs.kneighbors(np.array(vertices))   # [N, 8]
    distances = np.square(distances)
    d = np.max(distances, axis=1, keepdims=True)   # [N, 8]
    probs = np.exp(-distances/d)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    new_vertices = []
    for prob, index in zip(probs, indices):
        neighbors = np.array([vertices[ind] for ind in index])  # [8, 3]
        vertex = np.sum(neighbors * prob.reshape(-1, 1), axis=0)
        new_vertices.append(vertex)
    return new_vertices, atoms


def test_surface_file(input_file, surface_path, new_surface_path):
    lines = open(input_file, "r", encoding="utf-8").readlines()

    for i in range(0, len(lines), 2):
        name = lines[i].strip()[1: 5]

        surface_file = os.path.join(surface_path, "pdb{}.vert".format(name))
        new_surface_file = os.path.join(new_surface_path, "{}.vert".format(name))
        if not os.path.exists(surface_file):
            continue

        new_vertices, atoms = extract_vertix(surface_file)
        if len(new_vertices) <= 1:
            continue
        fw = open(new_surface_file, "w", encoding="utf-8")
        for vertex, atom in zip(new_vertices, atoms):
            line = " ".join([str(vertex[0]), str(vertex[1]), str(vertex[2])])
            line = line + " " + atom
            fw.write(line + "\n")
        fw.close()


def count_vertex_for_protein(file_path, seq_file):
    lines = open(seq_file, "r", encoding="utf-8").readlines()

    max_len_ver, max_len_protein = -1, -1
    rate = []
    for i in range(0, len(lines), 2):
        name = lines[i].strip()[1: 5]
        protein = lines[i+1].strip()
        protein_length = len(protein)
        max_len_protein = max(protein_length, max_len_protein)

        surface_file = os.path.join(file_path, "pdb{}.vert".format(name))
        if not os.path.exists(surface_file):
            continue
        len_vertices = len(open(surface_file, "r").readlines())
        max_len_ver = max(len_vertices, max_len_ver)
        rate.append(float(len_vertices) / protein_length)
    print(np.average(np.array(rate)))
    print(max_len_protein)
    print(max_len_ver)


def fill_idx(xyz, x, y, z, empty_index):

    for i in range(len(xyz)):

        for j in range(len(x)):

            if xyz[i,0] >= x[j] and xyz[i,0] <= x[j+1]:

                for k in range(len(y)):

                    if xyz[i,1] >= y[k] and xyz[i,1] <= y[k+1]:

                        for l in range(len(z)):

                            if xyz[i,2] >= z[l] and xyz[i,2] <= z[l+1]:

                                empty_index[i,0] = j+1
                                empty_index[i,1] = k+1
                                empty_index[i,2] = l+1

                                break
                        break
                break
    return empty_index


def merge_idx(idx):
    a = idx[:, 0] * (10 ** (np.log10(idx[:, 1]).astype(int) + 1)) + idx[:, 1]

    b = a * (10 ** (np.log10(idx[:, 2]).astype(int) + 1)) + idx[:, 2]

    return b


def octree(xyz):
    xyzmin = np.min(xyz, axis=0)
    xyzmax = np.max(xyz, axis=0)
    n = 0
    idx = np.zeros_like(xyz, dtype=int)

    number = 0
    #: there will be implemented more conditions to stop the split process
    while number < 4:
        x = np.linspace(xyzmin[0], xyzmax[0], n)
        y = np.linspace(xyzmin[1], xyzmax[1], n)
        z = np.linspace(xyzmin[2], xyzmax[2], n)
        idx = fill_idx(xyz, x, y, z, idx)

        n = (2 ** n) + 1
        number += 1
    idx = merge_idx(idx)
    return idx


def get_voxel_dict(ids, lines):
    voxel_dict = dict()
    for ind, line in zip(ids, lines):
        if ind not in voxel_dict:
            voxel_dict[ind] = []
        voxel_dict[ind].append(line)
    return voxel_dict


def normalize_coordinates(file_path, new_path):
    files = os.listdir(file_path)

    for name in files:
        input_file = os.path.join(file_path, name)
        lines = open(input_file, "r", encoding="utf-8").readlines()
        vertices, atoms = [], []
        for line in lines:
            phrases = line.strip().split()
            vertices.append([float(phrases[0]), float(phrases[1]), float(phrases[2])])
            atoms.append(phrases[-1])

        vertices = np.array(vertices)
        center = np.mean(vertices, axis=0)
        max_ = np.max(vertices, axis=0)
        min_ = np.min(vertices, axis=0)
        length = np.max(max_ - min_)
        vertices = (vertices - center) / length

        new_file = os.path.join(new_path, name)
        fw = open(new_file, "w", encoding="utf-8")
        for vertex, atom in zip(vertices, atoms):
            line = " ".join([str(term) for term in vertex])
            line = line + " " + atom
            fw.write(line + "\n")
        fw.close()


def down_sampling(file_path, new_path):
    files = os.listdir(file_path)

    for name in files:
        input_file = os.path.join(file_path, name)
        lines = open(input_file, "r", encoding="utf-8").readlines()

        if len(lines) <= 5000:
            os.system("cp %s %s" % (input_file, new_path))
        else:
            new_file = os.path.join(new_path, name)
            vertices, all_lines = [], []
            for line in lines:
                phrases = line.strip().split()
                coor = np.array([float(phrases[0]), float(phrases[1]), float(phrases[2])])
                if np.any(np.isnan(coor)):
                    continue
                vertices.append(coor)
                all_lines.append(line)

            vertices = np.array(vertices)
            ids = octree(np.array(vertices))
            voxel_dict = get_voxel_dict(ids, all_lines)
            total_points = len(all_lines)
            ratios = min(1.0, float(5000) / total_points)
            fw = open(new_file, "w", encoding="utf-8")
            for key in voxel_dict:
                points = voxel_dict[key]
                number = int(len(points) * ratios)
                samples = random.sample(points, number)
                for sample in samples:
                    fw.write(sample)
            fw.close()


def get_coor_atom(input_file):
    lines = open(input_file, "r", encoding="utf-8").readlines()
    if len(lines) == 0:
        return "", ""

    vertex_dict = dict()
    for line in lines:
        phrases = line.strip().split()
        aa = phrases[-1].split("_")[1]
        index = int(phrases[-1].split("_")[2])

        if aa not in amino_acid_dict:
            continue
        coor = np.array([float(phrases[0]), float(phrases[1]), float(phrases[2])])
        if np.any(np.isnan(coor)):
            continue
        vertex_dict[line] = index

    a = sorted(vertex_dict.items(), key=lambda x: x[1])
    # new_lines = []
    # coors = np.array(coors)
    #
    # while True:
    #     new_lines.append(vert_lines[tag])
    #     coor = np.reshape(coors[tag], (1, 3))
    #     aa = aas[tag]
    #
    #     del vert_lines[tag]
    #     del aas[tag]
    #     coors = np.delete(coors, tag, axis=0)
    #     if len(vert_lines) == 0:
    #         break
    #
    #     dist = np.sum(np.square(coor - coors), axis=-1)
    #     print(np.size(dist))
    #     inds = np.argsort(dist)[: min(len(dist), 30)]
    #     print(np.size(inds))
    #     dist2 = np.array([dist_dict[amino_acid_dict[aa]][amino_acid_dict[aas[j]]] for j in inds])
    #     tag = inds[np.argsort(dist2)[0]]

    coor = []
    atom = []
    for term in a:
        line, index = term[0], term[1]
        phrases = line.strip().split()
        coor.extend(phrases[: 3])
        atom.append(phrases[-1])

        # atom.append(phrases[-1].split("_")[0])
    coor = " ".join(coor)
    atom = " ".join(atom)
    return coor, atom


def get_coor_atom_heuristic(input_file):
    lines = open(input_file, "r", encoding="utf-8").readlines()
    if len(lines) == 0:
        return "", ""

    vert_dict = {}
    for line in lines:
        phrases = line.strip().split()
        aa = phrases[-1].split("_")[1]
        index = int(phrases[-1].split("_")[2])

        if aa not in amino_acid_dict:
            continue
        coor = np.array([float(phrases[0]), float(phrases[1]), float(phrases[2])])
        if np.any(np.isnan(coor)):
            continue
        vert_dict[line] = index

    new_lines = sorted(vert_dict.items(), key=lambda item: item[1])
    coor = []
    atom = []
    for item in new_lines:
        line = item[0]
        phrases = line.strip().split()
        coor.extend(phrases[: 3])
        atom.append(phrases[-1])

        # atom.append(phrases[-1].split("_")[0])
    coor = " ".join(coor)
    atom = " ".join(atom)
    return coor, atom


def extract_feature(input_file, output_seq, output_atom, output_coor, output_pdb, surface_path):
    lines = open(input_file, "r", encoding="utf-8").readlines()
    fw_seq = open(output_seq, "w", encoding="utf-8")
    fw_atom = open(output_atom, "w", encoding="utf-8")
    fw_coor = open(output_coor, "w", encoding="utf-8")
    fw_pdb = open(output_pdb, "w", encoding="utf-8")

    for i in range(0, len(lines), 2):
        protein = lines[i+1]
        name = lines[i].strip()[1: 5]
        surface_file = os.path.join(surface_path, "{}.vert".format(name))

        if protein.strip() == "":
            continue
        if not os.path.exists(surface_file):
            continue
        coor, atom = get_coor_atom_heuristic(surface_file, dist_dict)
        if coor == "" or atom == "" or protein.strip() == "":
            continue
        if len(coor.split()) <= 3:
            continue
        fw_seq.write(protein)
        fw_atom.write(atom + "\n")
        fw_coor.write(coor + "\n")
        fw_pdb.write(lines[i])
    fw_coor.close()
    fw_seq.close()
    fw_atom.close()
    fw_pdb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    FLAGS, _ = parser.parse_known_args()
    split = FLAGS.split
    file_path = FLAGS.data_path
    output_path = FLAGS.output_path

    surface_path = os.path.join(file_path, "msms")
    os.system("mkdir -p {}".format(surface_path))
    os.system("mkdir -p {}/{}".format(file_path, "msms_smooth"))
    os.system("mkdir -p {}/{}".format(file_path, "normalize_msms_smooth"))
    os.system("mkdir -p {}/{}".format(file_path, "octree_surface"))

    # smoothing, fasta file for pdb & seq, cath42_data for vert file obtained from MSMS
    input_file = os.path.join(file_path, "{}.fasta.txt".format(split))
    test_surface_file(input_file, surface_path, os.path.join(file_path, "msms_smooth"))

    # normalize
    normalize_coordinates(os.path.join(file_path, "msms_smooth"), os.path.join(file_path, "normalize_msms_smooth"))

    # down sampling
    down_sampling(os.path.join(file_path, "normalize_msms_smooth"), os.path.join(file_path, "octree_surface"))

    output_seq = os.path.join(output_path, "{}.seq.txt".format(split))
    out_coor = os.path.join(output_path, "{}.coor.txt".format(split))
    out_atom = os.path.join(output_path, "{}.atom.txt".format(split))
    out_pdb = os.path.join(output_path, "{}.pdb.txt".format(split))
    surface_path = os.path.join(file_path, "octree_surface")
    extract_feature(input_file, output_seq, out_atom, out_coor, out_pdb, surface_path)













