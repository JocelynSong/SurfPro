#!/bin/bash

data_path=cath42/octree_aa_surf_5k_sorted

output_path=cath_model_path

python3 fairseq_cli/train.py ${data_path} \
--save-dir ${output_path} \
--task fragment_protein_design \
--dataset-impl-source "ss" \
--criterion cross_entropy \
--arch surface_protein_model_base \
--encoder-embed-dim 256 \
--decoder-embed-dim 256 \
--gnn-layer 3 \
--decoder-layers 3 \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 5e-4 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-8' --warmup-updates 4000 \
--warmup-init-lr '5e-5' \
--weight-decay 0.0001 \
--clip-norm 0.01 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-sentences 4 \
--update-freq 1 \
--max-update 2000000 \
--max-epoch 100 \
--valid-subset validation \
--max-sentences-valid 4 \
--validate-interval 1 \
--save-interval 1 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--keep-interval-updates	10 \
--skip-invalid-size-inputs-valid-test
