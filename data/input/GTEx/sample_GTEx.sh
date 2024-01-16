#!/bin/sh
{ head -n 1 GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct; echo "5000	11688"; sed -n '3p' GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct; tail -n +4 GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct | shuf -n 5000; } > GTEx_v7_tpm_first5k_rand.gct
