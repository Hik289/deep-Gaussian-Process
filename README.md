# Deep Gaussian Process Experiments

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-pending-0E7C66.svg" alt="License"></a> <a href="#paper-or-reference"><img src="https://img.shields.io/badge/paper-reference-1F4E79.svg" alt="Paper or reference"></a> <img src="https://img.shields.io/badge/language-Python-3776AB.svg" alt="Python">
</p>

<p align="center">
  <strong>Deep GP notebooks and random-feature implementations for scalable uncertainty modeling.</strong>
</p>

<p align="center">
  <img src="assets/readme-figure.png" alt="Deep Gaussian Process Experiments overview" width="100%">
</p>

## Abstract

This repository is a conference-style artifact for deep Gaussian processes and random-feature approximations. It packages the code and notes needed to inspect the central research question: How can deep GP uncertainty be explored with scalable random-feature implementations? The emphasis is on transparent entry points, reproducible execution, and clear separation between code, local data, and generated outputs.

## Artifact at a Glance

| Item | Details |
| --- | --- |
| Research question | How can deep GP uncertainty be explored with scalable random-feature implementations? |
| Primary artifact | Deep GP notebooks and random-feature code folders. |
| Main entry points | `example_simple.ipynb`, `example.ipynb`, `deep_gp_random_features/` |
| Expected outputs | Posterior curves, uncertainty bands, and approximation diagnostics. |

## Repository Structure

| Item | Details |
| --- | --- |
| `deep_gp_random_features/` | random-feature code for scalable deep GP experiments. |
| `deep GP code/` | earlier deep GP implementation notes and scripts. |
| `PyDeepGP/` | included implementation reference used by the experiments. |
| `example.ipynb`, `example_simple.ipynb`, `GP.ipynb` | notebook entry points for reproducing core examples. |

## Reproducibility Protocol

1. `git clone git@github.com:Hik289/deep-Gaussian-Process.git`
2. `python -m venv .venv && source .venv/bin/activate`
3. `python -m pip install -U pip jupyter numpy scipy matplotlib scikit-learn`
4. Open `example_simple.ipynb` first, then move to the deeper experiments once the base environment runs.
5. Record the data window, random seed, software versions, machine type, and exact command used for any full rerun.
6. Store regenerated figures, tables, checkpoints, or reports under the existing result folders instead of overwriting raw inputs.

## Evaluation Protocol

| Step | Reviewer-facing check |
| --- | --- |
| Environment | Confirm the listed runtime or notebook environment starts without modifying tracked files. |
| Minimal run | Execute the smallest entry point before launching longer experiments. |
| Output check | Compare regenerated files with the expected figures, tables, logs, or reports named in this README. |
| Extension check | Add new runs as separate scripts, notebooks, or result folders with explicit names. |

## Expected Results

- The main scripts or notebooks should regenerate the project-specific artifacts listed in **Artifact at a Glance**.
- Outputs should be traceable to a command, parameter setting, and data window.
- Any private data path or machine-specific setting should be documented before sharing the artifact externally.

## Paper or Reference

No external paper link is currently attached to this project. For now, the code, notebooks, and notes in this repository are the primary reference artifact.

## Citation

If this repository supports a paper, cite the paper first and the artifact version second. If no paper is attached, cite the repository snapshot used in the experiment.

```bibtex
@misc{deep_gaussian_process_artifact_2026,
  title = {{Deep Gaussian Process Experiments}},
  author = {Hik289},
  year = {2026},
  howpublished = {\url{https://github.com/Hik289/deep-Gaussian-Process}},
  note = {Conference-style research artifact}
}
```

## License

No explicit license file is included yet. Add one before public reuse, redistribution, or package release.

## Reviewer Checklist

| Claim | How to inspect it |
| --- | --- |
| Code availability | Code and notebooks are present in the repository. |
| Reproducibility | The protocol above gives the expected setup and run order. |
| Result traceability | Generated outputs should live in named result, report, log, or output folders. |
| Extensibility | New experiments should preserve existing artifacts and add clearly named outputs. |
