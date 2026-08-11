# Adversarial Robustness of Deep Learning-Based ICS Anomaly Detectors

Code and frozen experiment outputs accompanying the manuscript
*"Why Architecture Matters: Adversarial Vulnerability Mechanisms in Deep Learning-Based ICS Anomaly Detectors"* (under review).

This repository is anonymized for peer review.

## What is here

```
scripts/     Experiment pipeline (training, attacks, defenses, gradient analysis)
  models.py                        Five detector architectures (LSTM-AE, USAD, GDN, TranAD, DAGMM)
  data_loader.py                   SWaT / WADI / SMD loading, normalization, windowing
  run_experiment.py                Main benchmark runner (baselines, attacks, AT defense)
  run_gradient_analysis.py         Gradient norm computation
  exp_a_gradient_within_dataset.py Within-dataset gradient-vs-ASR analysis
  verify_pipeline.py               Pipeline smoke test (5 detectors)
  config*.yaml                     Exact configurations used for every experiment batch
analysis/    Post-processing (no GPU required)
  analyze_results.py               Regenerates every figure and summary table from results/
  gen_fig1_architecture.py         Figure 1 (framework diagram)
results/     Frozen experiment outputs (aggregate + per-seed JSON)
             Do NOT hand-edit. Every table and figure in the manuscript is
             regenerated from these files by analysis/analyze_results.py.
```

## Reproducing the reported tables and figures (no GPU required)

```bash
pip install -r requirements.txt
python analysis/analyze_results.py --results-dir results --figures-dir figures
```

This regenerates every figure and prints every summary table reported in the
manuscript directly from the frozen outputs in `results/`.

## Re-running the experiments (GPU required)

Experiments were run on a single NVIDIA RTX 4090 (24 GB), Python 3.10,
PyTorch 2.1, IBM ART v1.16. Datasets are not redistributed here (see below).

```bash
cd scripts
python verify_pipeline.py                                          # smoke test
python run_experiment.py --config config.yaml          --dataset swat   # main benchmark (SWaT)
python run_experiment.py --config config_exp_b_smd.yaml --dataset smd   # SMD cross-dataset
python run_experiment.py --config config_exp_c_w50.yaml --dataset swat  # window w=50 (w100/w200 analogous)
python run_gradient_analysis.py                                    # gradient norms
```

The frozen threshold-sweep outputs (`results/threshold_sweep_swat.json`) were
produced by an earlier revision of the experiment driver and are included as
data; all other results reproduce through the commands above.

Seeds are fixed in the config files (`seeds: [42, 123, 456]`; the seed-sensitivity
batch additionally uses 7 and 2024).

## Datasets (not redistributed)

- **SWaT** — request access from the iTrust Centre for Research in Cyber
  Security, Singapore University of Technology and Design:
  https://itrust.sutd.edu.sg/itrust-labs_datasets/
  Place `SWaT_Dataset_Normal_v1.csv` and `SWaT_Dataset_Attack_v0.csv` under `data/swat/`.
- **SMD (Server Machine Dataset)** — public, via the OmniAnomaly repository:
  https://github.com/NetManAIOps/OmniAnomaly
  Place `ServerMachineDataset/` under `data/smd/`. The paper uses machine-1-1.
- The loader also supports **WADI** (same iTrust portal); WADI is not used in
  the current manuscript.

## License

MIT — see LICENSE.
