# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import shutil
import struct
from functools import lru_cache
import json
import numpy as np

import numpy as np
import torch
from fairseq.dataclass.constants import DATASET_IMPL_CHOICES
from fairseq.data.fasta_dataset import FastaDataset
from fairseq.file_io import PathManager
from fairseq.data.huffman import HuffmanMMapIndexedDataset, HuffmanMMapIndex

from . import FairseqDataset

from typing import Union


amino_acid_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                   "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                   "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def best_fitting_int_dtype(
    max_int_to_represent,
) -> Union[np.uint16, np.uint32, np.int64]:

    if max_int_to_represent is None:
        return np.uint32  # Safe guess
    elif max_int_to_represent < 65500:
        return np.uint16
    elif max_int_to_represent < 4294967295:
        return np.uint32
    else:
        return np.int64
        # we avoid np.uint64 because it doesn't save space and its type promotion behaves unexpectedly
        # https://github.com/numpy/numpy/issues/5745


def get_available_dataset_impl():
    return list(map(str, DATASET_IMPL_CHOICES))


def infer_dataset_impl(path):
    if IndexedRawTextDataset.exists(path):
        return "raw"
    elif IndexedDataset.exists(path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return "cached"
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return "mmap"
            elif magic == HuffmanMMapIndex._HDR_MAGIC[:8]:
                return "huffman"
            else:
                return None
    elif FastaDataset.exists(path):
        return "fasta"
    else:
        return None


def make_builder(out_file, impl, vocab_size=None):
    if impl == "mmap":
        return MMapIndexedDatasetBuilder(
            out_file, dtype=best_fitting_int_dtype(vocab_size)
        )
    elif impl == "fasta":
        raise NotImplementedError
    elif impl == "huffman":
        raise ValueError("Use HuffmanCodeBuilder directly as it has a different interface.")
    else:
        return IndexedDatasetBuilder(out_file)


def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None, source=True, sizes=None, motif_list=None,
                 epoch=1, train=True, protein_task=None, split="train"):
    if impl == "raw" and IndexedRawTextDataset.exists(path):
        assert dictionary is not None
        return IndexedRawTextDataset(path, dictionary, source=source)
    elif impl == "coor" and CoordinateDataset.exists(path):
        return CoordinateDataset(path, motif_list)
    elif impl == "label" and LabelDataset.exists(path):
        return LabelDataset(path)
    elif impl == "motif" and ProteinMotifDataset.exists(path):
        return ProteinMotifDataset(path, sizes, epoch, train)
    elif impl == "lazy" and IndexedDataset.exists(path):
        return IndexedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == "cached" and IndexedDataset.exists(path):
        return IndexedCachedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == "mmap" and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path)
    elif impl == "fasta" and FastaDataset.exists(path):
        from fairseq.data.fasta_dataset import EncodedFastaDataset
        return EncodedFastaDataset(path, dictionary)
    elif impl == "huffman" and HuffmanMMapIndexedDataset.exists(path):
        return HuffmanMMapIndexedDataset(path)
    elif impl == "segment" and FragmentProteinDataset.exists(path):
        return FragmentProteinDataset(path, dictionary, source=source, protein_task=protein_task)
    elif impl == "ss" and SecondaryStructureProteinDataset.exists(path):
        return SecondaryStructureProteinDataset(path, dictionary, source=source, split=split)
    elif impl == "geo" and GeometricSurfaceDataset.exists(path):
        return GeometricSurfaceDataset(path, source=source)
    elif impl == "chem_dist" and NeighborAtomDistSurfaceDataset.exists(path):
        return NeighborAtomDistSurfaceDataset(path, source=True)
    elif impl == "chem_atom" and NeighborAtomtypeSurfaceDataset.exists(path):
        return NeighborAtomtypeSurfaceDataset(path, source=True)
    elif impl == "surf_atom" and SurfaceAtomDataset.exists(path):
        return SurfaceAtomDataset(path, source=True)
    elif impl == "surf_aa" and SurfaceAminoAcidDataset.exists(path):
        return SurfaceAminoAcidDataset(path, source=True)
    elif impl == "surf_coor" and SurfaceAtomCoordinateDataset.exists(path):
        return SurfaceAtomCoordinateDataset(path, source=True)
    elif impl == "surf_identity" and SurfaceResidueIdentityDataset.exists(path):
        return SurfaceResidueIdentityDataset(path)
    elif impl == "control_coor" and ControlledCoordinateDataset.exists(path):
        return ControlledCoordinateDataset(path, split)
    elif impl == "control_identity" and ControlledIdentityDataset.exists(path):
        return ControlledIdentityDataset(path, split)
    elif impl == "attn" and AttentionDataset.exists(path):
        return AttentionDataset(path)
    return None


def dataset_exists(path, impl):
    if impl == "raw":
        return IndexedRawTextDataset.exists(path)
    elif impl == "mmap":
        return MMapIndexedDataset.exists(path)
    elif impl == "huffman":
        return HuffmanMMapIndexedDataset.exists(path)
    elif impl == "segment":
        return ProteinMotifDataset.exists(path)
    elif impl == "ss":
        return FragmentProteinDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


_code_to_dtype = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16,
    9: np.uint32,
    10: np.uint64,
}


def _dtype_header_code(dtype) -> int:
    for k in _code_to_dtype.keys():
        if _code_to_dtype[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


class IndexedDataset(FairseqDataset):
    """Loader for TorchNet IndexedDataset"""

    _HDR_MAGIC = b"TNTIDX\x00\x00"

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__()
        self.path = path
        self.fix_lua_indexing = fix_lua_indexing
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                "Index file doesn't match expected format. "
                "Make sure that --dataset-impl is configured properly."
            )
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dtype = _code_to_dtype[code]
            self._len, self.s = struct.unpack("<QQ", f.read(16))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), "rb", buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    @lru_cache(maxsize=8)
    def __getitem__(self, i) -> torch.Tensor:
        if not self.data_file:
            self.read_data(self.path)
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(index_file_path(path)) and PathManager.exists(
            data_file_path(path)
        )

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):
    def __init__(self, path, fix_lua_indexing=False):
        super().__init__(path, fix_lua_indexing=fix_lua_indexing)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx : ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        ptx = self.cache_index[i]
        np.copyto(a, self.cache[ptx : ptx + a.size])
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item


class IndexedRawTextDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, source=True, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                self.lines.append(line)
                if self.source:
                    prepend_bos = True
                else:
                    prepend_bos = False
                tokens = dictionary.encode_line(
                    line,
                    add_if_not_exist=False,
                    prepend_bos=prepend_bos,
                    append_eos=self.append_eos,
                    reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class CoordinateDataset(FairseqDataset):
    """
    Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, motif_list):
        self.coors_list = []
        self.lines = []
        self.sizes = []
        self.read_data(path, motif_list)
        self.size = len(self.coors_list)

    def read_data(self, path, motif_list):
        with open(path, "r", encoding="utf-8") as f:
            for ind, line in enumerate(f.readlines()):
                line = line.strip()
                self.lines.append(line)
                coors = line.split(",")
                protein_coor = []
                for i in range(0, len(coors), 3):
                    protein_coor.append([float(coors[i]), float(coors[i+1]), float(coors[i+2])])
                protein_coor = torch.tensor(np.array(protein_coor))
                mask = (motif_list[ind][1: -1] == 0).int().unsqueeze(-1)
                mean_coor = torch.sum(protein_coor * mask, dim=0) / mask.sum()
                protein_coor = protein_coor - mean_coor
                # mean_coor = torch.mean(protein_coor, dim=0)
                # protein_coor = protein_coor - mean_coor
                protein_coor = torch.cat([torch.tensor([[0, 0, 0]]), protein_coor, torch.tensor([[0, 0, 0]])], dim=0)
                self.coors_list.append(protein_coor)
                self.sizes.append(len(protein_coor))

        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.coors_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class ProteinMotifDataset(FairseqDataset):
    """
    Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dataset_sizes, epoch, train):
        self.motif_list = []
        self.sizes = []
        self.epoch = epoch
        self.read_data(path, dataset_sizes, self.epoch, train)
        self.size = len(self.motif_list)

    def read_data(self, path, dataset_sizes, epoch, train):
        with open(path, "r", encoding="utf-8") as f:
            for line, size in zip(f.readlines(), dataset_sizes):
                mask = np.ones(size)
                line = line.strip()
                indexes = line.split(",")
                if line != "":
                    indexes = [int(index)+1 for index in indexes]
                if train:
                    if epoch > 10:
                        epoch = 10
                    if len(indexes) < int(size*0.85):
                        diff = list(set(range(size)).difference(set(indexes)))
                        number = int(((int(size*0.85) - len(indexes)) / 10) * (10-epoch))
                        samples = random.sample(diff, number)
                        indexes.extend(samples)
                if line != "":
                    for ind in indexes:
                        mask[int(ind)] = 0
                mask[0] = 0
                mask[-1] = 0
                self.motif_list.append(torch.IntTensor(mask))
                self.sizes.append(len(mask))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.motif_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


def get_distribution(protein):
    dist = json.load(open("/mnt/data4/zhenqiaosong/protein_design/fragment_protein_design/data/distribution.json"))
    prob = dist[protein]
    return prob


class FragmentProteinDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, source=True, append_eos=True, reverse_order=False, protein_task=None):
        self.padding_idx = dictionary.pad()
        self.cls_idx = dictionary.bos()
        self.eos_idx = dictionary.eos()
        self.probability = get_distribution(protein_task)
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.append_eos = append_eos
        self.dictionary = dictionary
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def encode(self, words):
        ids = []
        for ind, word in enumerate(words):
            index = np.random.choice(range(20), p=self.probability[ind]) + 4
            ids.append(index)
        return ids

    def encode_line(
        self,
        line,
        prepend_bos=True,
        append_eos=True,
    ) -> torch.IntTensor:
        words = [aa for aa in line]

        nwords = len(words)
        ids = torch.IntTensor(nwords + int(prepend_bos) + int(append_eos))
        ids.fill_(self.padding_idx)

        if prepend_bos:
            ids[0] = self.cls_idx

        # seq_encoded = self.encode(words)
        seq_encoded = self.dictionary.encode(words)
        for idx, word in enumerate(seq_encoded):
            if word == 8:
                seq_encoded[idx] = 31

        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        ids[int(prepend_bos): len(seq_encoded) + int(prepend_bos)] = seq

        if append_eos:
            ids[len(seq_encoded) + int(prepend_bos)] = self.eos_idx

        return ids

    def read_data(self, path, dictionary):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                self.lines.append(line)

                if self.source:
                    tokens = self.encode_line(
                        line,
                        prepend_bos=True,
                        append_eos=self.append_eos,
                    ).long()

                else:
                    tokens = dictionary.encode_line(
                        line,
                        add_if_not_exist=False,
                        prepend_bos=True,
                        append_eos=self.append_eos,
                        reverse_order=self.reverse_order,
                    ).long()

                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class SecondaryStructureProteinDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, source=True, append_eos=True, reverse_order=False, split=None):
        self.padding_idx = dictionary.pad()
        self.cls_idx = dictionary.bos()
        self.eos_idx = dictionary.eos()
        # self.pdb_list = self.get_pdb_list(split)
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.append_eos = append_eos
        self.dictionary = dictionary
        self.reverse_order = reverse_order
        self.read_data(path)
        self.size = len(self.tokens_list)

    # def get_pdb_list(self, split):
    #     with open("/mnt/data4/zhenqiaosong/protein_design/data/CATH/data/cath_4_2/chain_set_splits.json") as f:
    #         dataset_splits = json.load(f)
    #     pdb_list = dataset_splits[split]
    #     return pdb_list

    def encode_target(
        self,
        line,
        prepend_bos=True,
        append_eos=True,
    ) -> torch.IntTensor:
        words = [aa for aa in line]

        nwords = len(words)
        ids = torch.IntTensor(nwords + int(prepend_bos) + int(append_eos))
        ids.fill_(self.padding_idx)

        if prepend_bos:
            ids[0] = self.cls_idx

        seq_encoded = self.dictionary.encode(words)

        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        ids[int(prepend_bos): len(seq_encoded) + int(prepend_bos)] = seq

        if append_eos:
            ids[len(seq_encoded) + int(prepend_bos)] = self.eos_idx

        return ids

    def read_data(self, path):
        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            # entry = json.loads(line)
            # name = entry['name']
            # if name not in self.pdb_list or name in ["1a1a.B", "4ga0.A", "1ut7.B", "5b7d.B", "2gpy.A", "2fsr.A", "2ia1.A", "2w5n.A"]:
            #     continue
            # seq = entry['seq']
            seq = line.strip()

            tokens = self.encode_target(
                seq,
                prepend_bos=True,
                append_eos=self.append_eos,
            ).long()

            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class SurfaceAtomDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, source=True):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.atom2index, self.index2atom = self.read_atom_dict()
        self.read_data(path)
        self.size = len(self.tokens_list)

    def get_original_atom(self, atom):
        for i in range(len(atom)):
            if atom[i].isdigit():
                return atom[: i]
        return atom

    # def read_atom_dict(self):
    #     atom_list = ["C", "CA", "N", "O", "CB", "<unk>"]
    #     atom2index = dict()
    #     index2atom = []
    #     for atom in atom_list:
    #         atom2index[atom] = len(atom2index)
    #         index2atom.append(atom)
    #     return atom2index, index2atom

    def read_atom_dict(self):
        lines = open("/mnt/data4/zhenqiaosong/protein_design/data/CATH/CATH_4_2/remove_unk/remove_repeat_id/extract_features/length_filtering/dict.txt", "r", encoding="utf-8").readlines()
        atom2index = dict()
        index2atom = []
        for line in lines:
            phrases = line.strip().split()
            atom = phrases[0].strip()
            # atom = self.get_original_atom(phrases[0].strip())

            if atom not in atom2index:
                atom2index[atom] = len(atom2index)
                index2atom.append(atom)
        return atom2index, index2atom

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                atoms = [word.strip().split("_")[0].strip() for word in words]
                # atoms = [self.get_original_atom(word.strip().split("_")[0].strip()) for word in words]
                tokens = []
                for word in atoms:
                    # if word in self.atom2index:
                    #     tokens.append(self.atom2index[word])
                    # else:
                    #     tokens.append(self.atom2index["<unk>"])
                    tokens.append(self.atom2index[word])

                self.tokens_list.append(torch.IntTensor(tokens))
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class SurfaceAminoAcidDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, source=True):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.aa2index = self.read_aa_dict()
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_aa_dict(self, ):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        aa2index = dict()
        for aa in alphabet:
            aa2index[aa] = len(aa2index)
        return aa2index

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                aas = [amino_acid_dict[word.strip().split("_")[-2].strip()] for word in words]
                tokens = []
                for word in aas:
                    tokens.append(self.aa2index[word])

                self.tokens_list.append(torch.IntTensor(tokens))
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class LabelDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                self.tokens_list.append(int(line.strip()))
                self.sizes.append(1)
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class GeometricSurfaceDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, source=True):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                # self.lines.append(line)
                tokens = []
                for i in range(0, len(words), 2):
                    mean = float(words[i].strip())
                    gaussian = float(words[i+1].strip())
                    if np.isnan(mean):
                        mean = 0.0
                    if np.isnan(gaussian):
                        gaussian = 0.0
                    tokens.extend([mean, gaussian])   # [1, L * 2]

                self.tokens_list.append(torch.tensor(tokens))
                self.sizes.append(int(len(tokens)/2))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class NeighborAtomDistSurfaceDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, source=True):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                tokens = []
                for i in range(0, len(words), 16):
                    this_words = [(1.0/float(word)) if float(word) != 0 else 1e-8 for word in words[i: i+16]]    #  [1, L * 16]
                    tokens.extend(this_words)

                self.tokens_list.append(torch.tensor(tokens))
                self.sizes.append(int(len(tokens)/16))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class NeighborAtomtypeSurfaceDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, source=True):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.atom2index, self.index2atom = self.read_atom_dict()
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_atom_dict(self, ):
        lines = open("/mnt/data4/zhenqiaosong/protein_design/data/CATH/CATH_4_2/final_training_data/atom_dict.txt", "r", encoding="utf-8").readlines()
        atom2index = dict()
        index2atom = []
        for line in lines:
            phrases = line.strip().split()
            atom = phrases[0].strip()
            number = int(phrases[1].strip())
            if number < 100:
                continue
            atom2index[atom] = len(atom2index)
            index2atom.append(atom)
        atom2index["<atom_unk>"] = len(atom2index)
        index2atom.append("<atom_unk>")
        return atom2index, index2atom

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                tokens = []
                for i in range(0, len(words), 16):
                    this_words = [self.atom2index[word] if word in self.atom2index else self.atom2index["<atom_unk>"] for word in words[i: i+16]]   # [1, L * 16]
                    tokens.extend(this_words)

                self.tokens_list.append(torch.IntTensor(tokens))
                self.sizes.append(int(len(tokens)/16))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class SurfaceAtomCoordinateDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, source=True):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                tokens = []
                for i in range(0, len(words), 3):
                    coor = [float(words[i].strip()), float(words[i+1].strip()), float(words[i+2].strip())]
                    tokens.extend(coor)   # [L * 3]

                self.tokens_list.append(torch.tensor(tokens))
                self.sizes.append(int(len(tokens)/3))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class AttentionDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.centers = []
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path):
        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            score = json.loads(line)["attn"]
            self.tokens_list.append(torch.tensor(score))   # [tgt, src]
            self.sizes.append(len(score))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class ControlledCoordinateDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, split="train"):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.centers = []
        self.split = split
        self.pdb_list = self.get_pdb_list(self.split)
        self.read_data(path)
        self.size = len(self.tokens_list)

    def get_pdb_list(self, split):
        with open("/mnt/data4/zhenqiaosong/protein_design/data/CATH/data/cath_4_2/chain_set_splits.json") as f:
            dataset_splits = json.load(f)
        pdb_list = dataset_splits[split]
        return pdb_list

    def read_data(self, path):
        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            entry = json.loads(line)
            name = entry['name']
            if name not in self.pdb_list or name in ["1a1a.B", "4ga0.A", "1ut7.B", "5b7d.B", "2gpy.A", "2fsr.A",
                                                     "2ia1.A", "2w5n.A"]:
                continue
            coords = entry['coords']
            for atom in ["N", "CA", "C", "O"]:
                for i, coor in enumerate(coords[atom]):
                    coords[atom][i] = np.array([float(coor[0]), float(coor[1]), float(coor[2])])

            this_coor = np.array(list(zip(
                coords['N'], coords['CA'], coords['C'], coords['O']
            )))    # [N, L, 4, 3]

            if np.isnan(this_coor).any():
                print(name)
            self.tokens_list.append(torch.tensor(this_coor))
            self.sizes.append(np.shape(this_coor)[1])
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class ControlledIdentityDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, split="train"):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.split = split
        self.pdb_list = self.get_pdb_list(self.split)
        self.read_data(path)
        self.size = len(self.tokens_list)

    def get_pdb_list(self, split):
        with open("/mnt/data4/zhenqiaosong/protein_design/data/CATH/data/cath_4_2/chain_set_splits.json") as f:
            dataset_splits = json.load(f)
        pdb_list = dataset_splits[split]
        return pdb_list

    def read_data(self, path):
        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            entry = json.loads(line)
            name = entry['name']
            if name not in self.pdb_list or name in ["1a1a.B", "4ga0.A", "1ut7.B", "5b7d.B", "2gpy.A", "2fsr.A", "2ia1.A", "2w5n.A"]:
                continue
            surf_id = entry['surf_id']
            surf_id[0] = 1
            surf_id[-1] = 1

            self.tokens_list.append(torch.tensor(surf_id))
            self.sizes.append(len(surf_id))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class SurfaceResidueIdentityDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                tokens = [0]
                for word in words:
                    tokens.append(int(word))
                tokens.append(0)  # L + 2

                tokens = torch.tensor(tokens, dtype=torch.int64)
                self.tokens_list.append(tokens)
                self.sizes.append(int(len(tokens)))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class IndexedDatasetBuilder:
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float32: 4,
        np.double: 8,
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, "wb")
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]

    def add_item(self, tensor):
        # +1 for Lua compatibility
        bytes = self.out_file.write(np.array(tensor.numpy() + 1, dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        with open(data_file_path(another_file), "rb") as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, "wb")
        index.write(b"TNTIDX\x00\x00")
        index.write(struct.pack("<Q", 1))
        index.write(
            struct.pack("<QQ", _dtype_header_code(self.dtype), self.element_size)
        )
        index.write(struct.pack("<QQ", len(self.data_offsets) - 1, len(self.sizes)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        index.close()


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer:
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", _dtype_header_code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = _code_to_dtype[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
        )
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return PathManager.exists(index_file_path(path)) and PathManager.exists(
            data_file_path(path)
        )


def get_indexed_dataset_to_local(path) -> str:
    local_index_path = PathManager.get_local_path(index_file_path(path))
    local_data_path = PathManager.get_local_path(data_file_path(path))

    assert local_index_path.endswith(".idx") and local_data_path.endswith(".bin"), (
        "PathManager.get_local_path does not return files with expected patterns: "
        f"{local_index_path} and {local_data_path}"
    )

    local_path = local_data_path[:-4]  # stripping surfix ".bin"
    assert local_path == local_index_path[:-4]  # stripping surfix ".idx"
    return local_path


class MMapIndexedDatasetBuilder:
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)
