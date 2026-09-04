# MoSim


[![IEEE MASCOTS](https://img.shields.io/badge/IEEE%20MASCOTS-2026-00629B)](https://mascots26.iitis.pl/)
[![License](https://img.shields.io/badge/License-MIT-3DA639)](LICENSE)

**Official artifact for the IEEE MASCOTS 2026 paper**<br>
*Accurate Simulation of Distributed Training Jobs with Network Contention Modeling*

Yeonho Yoo<sup>†</sup>, Hyunho Lee<sup>†</sup>, Hyunmok Choi<sup>†</sup>, Chuck Yoo<sup>&#42;</sup>, and Gyeongsik Yang<sup>&#42;</sup>

MoSim is a trace-driven GPU-cluster simulator that models how worker placement and shared
network interfaces affect distributed-training job completion times. This repository contains
the cluster-level simulator, prepared traces and profiles, physical-testbed results, and scripts
that reproduce Tables II–IV of the paper without requiring a GPU.

## Artifact scope

The bundled artifact reproduces Tables II, III, and IV. It does not include the ASTRA-sim
profile-generation workflow or the source measurements used for Fig. 2, Fig. 4, and Table V.

## Quick start

| Requirement | Version |
| --- | --- |
| Rust | 1.78+ |
| Python | 3.9+; standard library only for the reproduction scripts |
| OS | Linux or macOS |

From the repository root:

```bash
cargo build --release --locked --manifest-path mosim/Cargo.toml
bash scripts/run.sh
bash scripts/analyze.sh
```

The simulation and analysis normally finish in a few seconds after the Rust build. Generated
CSV and Markdown tables are written to `result/paper/tab{2,3,4}/out/`; `result/rawdata.csv`
contains the combined per-job outputs.

The paper uses the following artifact configurations. Tiresias and Pollux denote the baseline
configurations evaluated in the paper; their original implementations are not vendored here.

| Method | Single-job profile | Interference model |
| --- | --- | --- |
| Tiresias | real-GPU profiling | none |
| Pollux | real-GPU profiling | fixed 10% networking-time penalty |
| **MoSim** | ASTRA-sim | dynamic MoSim model |

To rerun experiments after changing parameters or prepared inputs, bypass cached outputs:

```bash
FORCE=1 FORCE_PREPARE=1 bash scripts/run.sh
```

See [`GUIDE.ipynb`](GUIDE.ipynb) for an executable walkthrough. Run
`python3 simulator-trace-timer-bw.py --help` for individual simulator options. Jupyter is only
required for the notebook, not for the reproduction scripts.

## Repository map

| Path | Purpose |
| --- | --- |
| `simulator-trace-timer-bw.py` | User-facing CLI |
| `mosim/src/gpu_job.rs` | Job model and bandwidth demand |
| `mosim/src/gpu_cluster.rs` | Placement and network-contention models |
| `mosim/src/gpu_scheduler.rs`, `mosim/src/timer.rs` | Scheduling and discrete-event execution |
| `scripts/run.sh`, `scripts/analyze.py` | Paper experiments and Tables II–IV |
| `data/` | Prepared traces, profiles, and physical-testbed results |

## Data provenance

- `data/single_job_simulation/` contains precomputed profiles generated using
  [ASTRA-sim](https://github.com/astra-sim/astra-sim). ASTRA-sim itself is not vendored in this
  repository. When using these profiles, please also cite Won et al., “ASTRA-sim2.0,” ISPASS
  2023 ([DOI](https://doi.org/10.1109/ISPASS57527.2023.00035)).
- `data/trace/` contains modified workload samples derived from the
  [Microsoft Philly trace](https://github.com/msr-fiddle/philly-traces) by Jeon et al. The
  source dataset is distributed under
  [CC BY 4.0](https://github.com/msr-fiddle/philly-traces/blob/master/LICENSE); these files
  select and transform records into MoSim's trace schema. Please cite
  [“Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads”](https://www.usenix.org/conference/atc19/presentation/jeon).
- `data/profiling/` and `data/testbed/` contain measurements collected on the authors' V100
  testbed.

## Citation

If you use MoSim, please cite the paper. GitHub also exposes the same metadata from
[`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{yoo2026accurate,
  author={Yoo, Yeonho and Lee, Hyunho and Choi, Hyunmok and Yoo, Chuck and Yang, Gyeongsik},
  title={Accurate Simulation of Distributed Training Jobs with Network Contention Modeling},
  booktitle={2026 34th IEEE International Symposium on Modeling, Analysis, and Simulation of Computer and Telecommunication Systems (MASCOTS)},
  year={2026},
  address={Genova, Italy},
  publisher={IEEE}
}
```

## License

The MoSim source code is licensed under the MIT License; see [`LICENSE`](LICENSE). Upstream tools
and datasets remain subject to their respective terms.

<details>
<summary><strong>Acknowledgments</strong></summary>

This research was supported by National Research Foundation of Korea (NRF) grant funded by
Korea government (MSIT) (RS-2024-00336564), by Institute of Information & Communications
Technology Planning & Evaluation (IITP) grant funded by Ministry of Science and ICT (MSIT)
(RS-2026-25518394), by ICT Creative Consilience Program through IITP grant funded by MSIT
(IITP-2026-RS-2020-II201819), by IITP under the Artificial Intelligence Convergence Innovation
Human Resources Development grant funded by Korea government (MSIT)
(IITP-2026-RS-2023-00254592), by ANCHOR through Seoul ANCHOR Center funded by MOE and Seoul
Metropolitan Government (2026-ANCHOR-01-003-09), and by computing support from Lambda Cloud.

</details>

## Contact

For artifact, code, or reproduction questions, [open an issue](https://github.com/OSSS-KU/MoSim/issues).
For questions about the paper, contact the corresponding authors, Chuck Yoo and Gyeongsik Yang.
