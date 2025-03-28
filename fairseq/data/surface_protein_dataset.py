# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def collate_features(value):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s[value] for s in samples]
        size = max(v.size(0) for v in values)

        batch_size = len(values)
        res = values[0].new(batch_size, size).fill_(0.0)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][: len(v)])
        return res

    def collate_target(
            value,
            pad_idx,
            pad_to_length=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s[value] for s in samples]
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)

        batch_size = len(values)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][: len(v)])
        return res

    def collate_prev_tokens(
            value,
            pad_idx,
            pad_to_length=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s[value] for s in samples]
        prev_tokens = [v[: -1] for v in values]
        targets = [v[1: ] for v in values]

        size = max(v.size(0) for v in prev_tokens)
        size = size if pad_to_length is None else max(size, pad_to_length)

        batch_size = len(values)
        res_prev = values[0].new(batch_size, size).fill_(pad_idx)
        res_target = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(prev_tokens):
            copy_tensor(v, res_prev[i][: len(v)])
        for i, v in enumerate(targets):
            copy_tensor(v, res_target[i][: len(v)])

        return res_prev, res_target

    id = torch.LongTensor([s["id"] for s in samples])
    # labels = torch.LongTensor([s["label"] for s in samples])
    coor_features = collate_features(
        "coor_feature",
    )
    aa_features = collate_target(
        "aa_feature",
        pad_idx,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    prev_tokens, target = collate_prev_tokens(
        "target",
        pad_idx,
        pad_to_length=pad_to_length["source"]
        if pad_to_length is not None
        else None,
    )
    # ground_truth = collate_target("target", pad_idx)
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["aa_feature"].size(0) for s in samples]
    )
    tgt_lengths = torch.LongTensor(
        [s["target"].size(0)-1 for s in samples]
    )
    # src_lengths, sort_order = src_lengths.sort(descending=True)
    # id = id.index_select(0, sort_order)
    # geo_features = geo_features.index_select(0, sort_order)
    # chem_dist_features = chem_dist_features.index_select(0, sort_order)
    # chem_atom_features = chem_atom_features.index_select(0, sort_order)
    # prev_tokens = prev_tokens.index_select(0, sort_order)
    # target = target.index_select(0, sort_order)
    ntokens = tgt_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {"coor_features": coor_features, "aa_features": aa_features, "src_lengths": src_lengths,
                      "prev_output_tokens": prev_tokens, "tgt_lengths": tgt_lengths},
        "target": {"aa": target},
    }
    # batch = {
    #     "id": id,
    #     "nsentences": len(samples),
    #     "ntokens": ntokens,
    #     "net_input": {"atom_features": atom_features, "coor_features": coor_features,
    #                   "src_lengths": src_lengths, "prev_output_tokens": prev_tokens,
    #                   "tgt_tokens": target}
    # }
    return batch


class SurfaceProteinDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        motif:
        motif_sizes
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src_dict=None,
        aas=None,
        aa_sizes=None,
        coors=None,
        coors_sizes=None,
        tgt=None,
        tgt_sizes=None,
        label_dataset=None,
        left_pad_source=False,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
    ):
        self.coor = coors
        self.aa = aas
        self.tgt = tgt
        # self.labels = label_dataset
        self.coor_sizes = np.array(coors_sizes)
        self.aa_sizes = np.array(aa_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.aa_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.aa_sizes
        )
        self.src_dict = src_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.pad_to_multiple = pad_to_multiple
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        aa_item = self.aa[index]
        coor_item = self.coor[index]
        tgt_item = self.tgt[index]
        # label = self.labels[index]

        example = {
            "id": index,
            "aa_feature": aa_item,
            "coor_feature": coor_item,
            "target": tgt_item,
        }
        return example

    def __len__(self):
        return len(self.tgt)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.geo_sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.aa_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.geo_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.aa_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.aa_sizes, self.tgt_sizes, indices, max_sizes,
        )
