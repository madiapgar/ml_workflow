#!/bin/bash

snakemake \
    -s workflow/snakefile \
    -c 7 \
    --use-conda \
    --keep-going \
    --configfile workflow/config_files/config.yml