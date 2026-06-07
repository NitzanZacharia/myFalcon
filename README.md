# Entity Binding Mechanisms Across Positional Embedding and Hybrid Architectures

This repository contains the code for our NLP seminar project, which extends the official codebase of the paper: "Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context" ([link](https://arxiv.org/abs/2510.06182)). 

While the original work focused exclusively on Transformer models utilizing RoPE positional embeddings, our project expands this interpretability framework to investigate alternative positional encodings and hybrid state-space architectures. 

### Our Additions & Extensions
Building upon the original interchange intervention framework, we have introduced several key additions:
* [cite_start]**Alternative Positional Encodings:** We extended the residual stream patching analysis to ALiBi models (MPT-7B, Bloomz-3B), confirming that the positional, lexical, and reflexive binding mechanisms generalize beyond RoPE[cite: 31, 32, 59].
* [cite_start]**Hybrid & State-Space Models (SSMs):** We provided the first analysis of entity binding in hybrid Mamba-Attention models (Falcon-H1-3B, Zamba2-2.7B) and pure Mamba models (Falcon-Mamba-7B)[cite: 34, 37, 59].
* [cite_start]**Multi-Layer Component Patching:** We introduced a novel patching methodology for hybrid models to isolate and evaluate the cumulative contributions of specific components (e.g., Attention vs. Mamba) across all layers simultaneously[cite: 35, 91, 92].
* [cite_start]**100-Entity Scalability Testing:** We scaled the original 20-entity binding task up to 100 bound entities to probe model capacity, representation collapse, and routing adaptations under heavy context loads[cite: 38, 129, 130].

<p align="center">
  <img width="864" height="830" alt="mechs_fig1" src="https://github.com/user-attachments/assets/e3ac9cdf-add7-4f02-96d0-f2b75e359651" />
</p>

### Files
The codebase includes the original framework alongside our new experimental pipelines:
* `CausalAbstraction/` - A copy of the official [CausalAbstraction](https://github.com/atticusg/CausalAbstraction) codebase, utilized for running the interchange interventions.
* `grammar/` - Directory defining all of our binding tasks (`schemas.py`) and the logic to convert them into a CausalModel (`task_to_causal_model.py`).
* `tasks/dist.py` - The primary script for running experiments, selecting counterfactuals, and targeting specific models.
* `training.py` - Setup code and counterfactual definitions required by `dist.py`.
* `plotting.py` - Scripts used for generating the layer-wise distribution and patching effect figures.
* `example.ipynb` - An out-of-the-box example script demonstrating the main interchange intervention and results plotting.

[cite_start]*Note: Additional scripts for our 100-entity evaluations and component patching can also be found at [https://github.com/NitzanZacharia/100rep.git](https://github.com/NitzanZacharia/100rep.git)[cite: 17].*

---

### Citation
If you utilize our extended architectures evaluation, please cite our work:
```bibtex
@misc{moryles2026entity,
    title={Entity Binding Mechanisms Across Positional Embedding and Hybrid Architectures},
    author={Inbal Moryles and Aviv Yossef and Nitzan Zacharia},
    year={2026}
}
