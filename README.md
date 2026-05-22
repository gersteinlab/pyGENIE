# pyGENIE

A Python library for programmatic access and analysis of [AACR Project GENIE](https://www.aacr.org/professionals/research/aacr-project-genie/) cancer genomics data.

## Installation

```bash
pip install pygenie-sdk
```

## Requirements

- Python ≥ 3.8
- pandas ≥ 1.3
- pyarrow (strongly recommended — enables 8x faster cross-session loading)

AACR Project GENIE data must be downloaded separately from [Synapse](https://www.synapse.org/) following registration and approval of a data use agreement.

## Quick Start

```python
from pygenie_sdk import GENIEClient

genie = GENIEClient("/path/to/genie/data", version="16.1-public")

# Build LUAD cohort and query mutations
luad = genie.build_cohort(oncotree_code="LUAD")
luad_muts = genie.query_mutations_by_cohort(oncotree_code="LUAD")

# Mutation landscape
genie.plot_top_mutated_genes(df=luad_muts, top_n=20)

# Clinical characterization
genie.plot_sex_distribution(df=luad_muts)
genie.plot_age_distribution(df=luad_muts)
genie.gene_cooccurrence_with_clinical("KRAS", cohort=luad)

# Survival analysis
genie.plot_age_survival(oncotree_code="LUAD", stratify_by="SEX")

# TMB and mutational signatures
genie.plot_mutation_burden(df=luad_muts, method="tmb")
genie.run_sigprofiler(output_dir="./results", oncotree_code="LUAD")
genie.plot_signature_exposure(sigprofiler_output_dir="./results")
```

## Features

- Clinical and genomic queries (mutations, fusions, CNA)
- Cohort construction and comparison
- Tumor mutational burden (TMB) estimation
- Co-mutation and mutual exclusivity analysis
- Mutational signature analysis via SigProfiler and SigMA
- Kaplan-Meier survival estimation
- Lollipop plots, mutation landscapes, and more

## Citation

Coming soon.

## License

MIT
