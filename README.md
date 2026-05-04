# Real-Time Neural Distributed Energy Resources Dispatch with Feasibility Guarantees

This repository contains the implementation of the paper **"Real-Time Neural Distributed Energy Resources Dispatch with Feasibility Guarantees"**.

## Overview

The code provides a neural network-based approach for real-time dispatch of distributed energy resources while ensuring system feasibility guarantees. The framework enables fast and reliable decision-making for modern power systems with high penetration of renewable energy sources.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `Solving_method/` | Main implementation files. Running the scripts in this folder will directly reproduce the results presented in the paper. |
| `System_data/` | System topology parameters and configuration files. You can view and modify these parameters to adapt the framework to different test systems. |
| `Data_generation/` | Scripts and tools for generating training data used in the neural network training process. |

## Getting Started

To reproduce the results from the paper:

1. Open the `33_bus` folder or the `129_bus` folder as the root directory.
2. Navigate to the `Solving_method` directory.
3. Run the provided scripts directly:
   - `NN_direct.py`
   - `NN_penalty.py`
   - `NN_penalty_oproj.py`
   - `NN_bisection.py`

All necessary dependencies and configurations are included in the respective folders.

## Customization

- **To modify system parameters:** Edit files in the `System_data` folder.
- **To regenerate or customize training data:** Use the scripts in the `Data_generation` folder.

## Citation

If you use this code in your research, please cite our paper:

J. Zhu, Y. Xu, and H. Sun, "Real-Time Neural Distributed Energy Resources Dispatch with Feasibility Guarantees," *arXiv preprint arXiv:2605.00317*, May 2026. [Online]. Available: https://arxiv.org/abs/2605.00317

### BibTeX

```bibtex
@article{zhu2026realtime,
  title={Real-Time Neural Distributed Energy Resources Dispatch with Feasibility Guarantees},
  author={Zhu, Jie and Xu, Yinliang and Sun, Hongbin},
  journal={arXiv preprint arXiv:2605.00317},
  year={2026},
  month={may},
  url={https://arxiv.org/abs/2605.00317}
}


