# Entity Binding Mechanisms Across Positional Embedding and Hybrid Architectures

This repository is an extension of the official code for the paper: **"Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context"** ([arXiv:2510.06182](https://arxiv.org/abs/2510.06182)).

<p align="center">
  <img width="864" height="830" alt="mechs_fig1" src="https://github.com/user-attachments/assets/e3ac9cdf-add7-4f02-96d0-f2b75e359651" />
</p>

The original work by Yoav Gur-Arieh, Mor Geva, and Atticus Geiger introduced a framework for studying how large language models retrieve bound entities in-context, distinguishing positional, lexical, and reflexive retrieval mechanisms. While the original work focused exclusively on Transformer models utilizing RoPE positional embeddings, our project expands this interpretability framework to investigate alternative positional encodings and hybrid state-space architectures. 

### Our Additions & Extensions
Building upon the original interchange intervention framework, we have introduced several key additions:

- **Alternative Positional Encodings:** We extended the residual stream patching analysis to ALiBi models (MPT-7B, Bloomz-3B), confirming that the positional, lexical, and reflexive binding mechanisms generalize beyond RoPE.
- **Hybrid & State-Space Models (SSMs):** We provided the first analysis of entity binding in hybrid Mamba-Attention models (Falcon-H1-3B, Zamba2-2.7B) and pure Mamba models (Falcon-Mamba-7B).

- **Multi-Layer Component Patching:** We introduced a novel patching methodology for hybrid models to isolate and evaluate the cumulative contributions of specific components (e.g., Attention vs. Mamba) across all layers simultaneously.

- **100-Entity Scalability Testing:** We scaled the original 20-entity binding task up to 100 bound entities to probe model capacity, representation collapse, and routing adaptations under heavy context loads.

<p align="center">
  <img width="864" height="830" alt="mechs_fig1" src="https://github.com/user-attachments/assets/e3ac9cdf-add7-4f02-96d0-f2b75e359651" />
</p>

### Files
The codebase includes the original framework alongside our new experimental pipelines:
* `CausalAbstraction/` - A copy of the official [CausalAbstraction](https://github.com/atticusg/CausalAbstraction) codebase with minor quality-of-life tweaks, utilized for running the interchange interventions.
* `grammar/` - Directory defining all of our binding tasks (`schemas.py`) and the logic to convert them into a CausalModel (`task_to_causal_model.py`).
* `tasks/dist.py` - The primary script for running experiments, selecting counterfactuals, and targeting specific models.
* `training.py` - Setup code and counterfactual definitions required by `dist.py`.
* `plotting.py` - Scripts used for generating the layer-wise distribution and patching effect figures.
* `example.ipynb` - An out-of-the-box example script demonstrating the main interchange intervention and results plotting.



---
## Original Citation

If you use this code, please cite the original paper:

```bibtex
@misc{gurarieh2025mixing,
    title={Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context},
    author={Yoav Gur-Arieh and Mor Geva and Atticus Geiger},
    year={2025},
    eprint={2510.06182},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
### Citation
If you utilize our extended architectures evaluation, please cite our work:
```bibtex
@misc{moryles2026entity,
    title={Entity Binding Mechanisms Across Positional Embedding and Hybrid Architectures},
    author={Inbal Moryles and Aviv Yossef and Nitzan Zacharia},
    year={2026}
}
```
