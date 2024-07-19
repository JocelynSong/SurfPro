#!/bin/bash

data_path=cath42/octree_aa_surf_5k_sorted

output_path=cath_model_path
generation_path=output/cath42_surface
mkdir -p ${generation_path}

python3 fairseq_cli/validate.py ${data_path} \
--task fragment_protein_design \
--dataset-impl-source "ss" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--eval-aa-recovery