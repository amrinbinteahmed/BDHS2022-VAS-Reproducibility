# BDHS 2022 Vitamin A Supplementation — Analysis Code

Python analysis code accompanying the study of vitamin A supplementation receipt among children aged 6–59 months in Bangladesh using BDHS 2022.

## Data access
The restricted DHS microdata are **not** included. See [DATA_ACCESS.md](DATA_ACCESS.md) for instructions on requesting the BDHS 2022 children recode file (BDKR81FL).

## Software
The manuscript reports Python 3.13.9 (Anaconda distribution), 64-bit Windows 11. Main package versions are listed in [requirements.txt](requirements.txt).

## Running the analysis
1. Obtain authorized BDHS 2022 KR microdata from the DHS Program.
2. Install the relevant Python packages in a compatible environment.
3. Review each script's input paths and required intermediate files, and adjust them for your local directory.
4. Execute the relevant scripts in analytical order, checking intermediate outputs and manuscript specifications.

**Scope and limitations:** The scripts reflect analysis stages and some historical development or audit versions. They are not represented as a single fully tested, one-command reproduction pipeline. Do not assume every script has been end-to-end verified on a fresh machine. The manuscript and its supplementary materials specify the reported analyses and final model settings.

## Reproducibility
The manuscript documents sample selection, variable coding, training-only preprocessing, PSU-aware validation, model tuning and seeds. The selected final model was XGBoost; ANN was evaluated as a comparator. See the Methods and Supplementary Tables S1–S2 for details.

## Data protection
Do not commit DHS microdata, row-level predictions, credentials, or private participant-level outputs to this public repository.
