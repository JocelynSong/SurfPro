#!/bin/bash

data_path=enzyme_design/Enzyme_Design_Data

save_path=cath_model_path
output_path=${save_path}/finetune_enzyme_design
generation_path=output/enzyme_output
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