# Escaping Preference Collapse: Cycle Dynamics in Nash Learning from Human Feedback

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)


> **Official codebase, PyTorch simulations, and interactive notebooks for analyzing algorithmic instability and phase transitions in Large Language Model (LLM) alignment.**

## Motivation and Background
Standard Reinforcement Learning from Human Feedback (RLHF) optimizes a single scalar reward model. However, human preferences frequently exhibit non-transitive structures, mathematically formalized as Condorcet cycles (e.g., $A \succ B \succ C \succ A$). When optimizing a scalar reward model against such cyclic preferences, RLHF suffers from **preference collapse**, resulting in an impoverished policy that fails to capture the true distributional diversity of human intent.

Nash Learning from Human Feedback (NLHF) addresses this limitation by formulating alignment as a two-player zero-sum game, allowing the model to converge to a mixed strategy. However, applying iterative learning dynamics to non-transitive preference data introduces severe algorithmic instability, frequently causing the policy to oscillate endlessly in limit cycles. 

This repository provides the mathematical framework and empirical simulations to analyze, bound, and stabilize these cycle dynamics.

---

## Phase Transitions in Algorithmic Drift
We prove that applying regularized Mirror Descent (Nash-MD) to cyclic preferences undergoes a strict three-phase transition. This transition is governed mathematically by the ratio of the learning rate ($\eta$) to the KL-regularization parameter ($\tau$).

![Nash-MD Phase Transition Dynamics](./assets/phase_transition.gif)

* **Expansive Regime ($\tau \ll \eta$):** Discretization error strictly dominates, injecting fictitious energy into the system. The policy diverges toward the simplex boundary, triggering preference collapse.
* **Stable Limit Cycle ($\tau \propto \eta$):** The expansive discretization error and the dissipative regularization balance precisely, sustaining a stable, non-zero limit cycle.
* **Collapsing Regime ($\tau \gg \eta$):** Over-regularization dominates, monotonically contracting the cycle into the exact stationary Nash Equilibrium.

---

## Key Findings and Visualizations

### 1. Initial Cycle Distributions (Lemma 2)
We mathematically bound the initial conditions of the system, proving that while uniform initialization guarantees cyclic behavior, the probability of initializing into a highly divergent, high-energy orbit decays exponentially.
<br>
<img src="./assets/lemma2_distribution.pdf" width="600" alt="Lemma 2 Distribution">

### 2. Algorithmic Phase Transitions (Theorem 1)
Tracking the "excess energy" of the policy over discrete iterations empirically validates the precise mathematical thresholds where cycles expand, stabilize, or collapse.
<br>
<img src="./assets/theorem1_phasetransition.pdf" width="600" alt="Theorem 1 Phase Transition">

### 3. The Stochasticity Gap
We formalize the impact of mini-batch gradient noise. Stochasticity injects massive expansive variance into the discrete drift, requiring a mathematically precise increase in the KL penalty to maintain cycle stability.
<br>
<img src="./assets/stochastic_drift.pdf" width="600" alt="Stochastic Drift">

### 4. High-Dimensionality and Neural Networks
We prove that these continuous-time topological dynamics scale robustly. The identified phase transitions hold in high-dimensional ($N=50$) random tournaments and within the loss landscapes of parameterized PyTorch neural networks.
<br>
<img src="./assets/n_dim_drift.pdf" width="400" alt="N-Dimensional Drift"> <img src="./assets/nn_drift.pdf" width="400" alt="Neural Network Drift">

---

## Repository Structure

```text
├── assets/
│   ├── phase_transition.gif
│   ├── lemma2_distribution.pdf
│   ├── theorem1_phasetransition.pdf
│   ├── stochastic_drift.pdf
│   ├── n_dim_drift.pdf
│   └── nn_drift.pdf
├── notebooks/
│   ├── 01_initial_distribution.ipynb      # Verifying the O(e^{-C/3}) tail
│   ├── 02_tabular_phase_transition.ipynb  # Simulating discrete Nash-MD drift
│   ├── 03_stochastic_batching.ipynb       # Mini-batch variance analysis
│   └── 04_neural_network_validation.ipynb # PyTorch policy parameterization
├── scripts/
│   ├── nlhf_simulations.py                # Core mathematical simulations
│   ├── nlhf_advanced_sims.py              # High-dim & Neural Network sims
│   └── generate_animations.py             # Script to render the simplex GIF
├── requirements.txt
└── README.md
```

---

## Getting Started

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/escaping-preference-collapse.git](https://github.com/yourusername/escaping-preference-collapse.git)
cd escaping-preference-collapse
pip install -r requirements.txt
```

To run the interactive Jupyter notebooks and replicate the empirical findings:
```bash
jupyter notebook
```

To locally regenerate the phase transition animation:
```bash
python scripts/generate_animations.py
```

---


## Acknowledgments
The author gratefully acknowledges Prof. Weijie J. Su (University of Pennsylvania) and Prof. Swagatam Das (Indian Statistical Institute) for their foundational insights into the Condorcet Paradox, the statistical limits of reward models, and game-theoretic alignment.
