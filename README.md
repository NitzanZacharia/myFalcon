# Mixing Mechanisms: Extended — How Language Models Retrieve Bound Entities In-Context

This repository is an extension of the official code for the paper: **"Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context"** ([arXiv:2510.06182](https://arxiv.org/abs/2510.06182)).

<p align="center">
  <img width="864" height="830" alt="mechs_fig1" src="https://github.com/user-attachments/assets/e3ac9cdf-add7-4f02-96d0-f2b75e359651" />
</p>

The original work by Yoav Gur-Arieh, Mor Geva, and Atticus Geiger introduced a framework for studying how large language models retrieve bound entities in-context, distinguishing positional, lexical, and reflexive retrieval mechanisms. This extension builds on that foundation with additional models, infrastructure, and experimental tooling.

---

## What's New in This Extension

### New Model Support

**`binding_model_wrappers.py`** — An abstract `BindingModelWrapper` class and concrete implementations for model families not covered in the original paper:
- `MPTModelWrapper` — MosaicML MPT instruct models, with custom prompt formatting for their instruction format.
- `MambaModelWrapper` — A unified wrapper supporting all Mamba-family variants (Falcon3, OpenHermes, Codestral, Mamba2, Zamba), with auto-detection of chat templates.
- `create_model_wrapper()` — Factory function for selecting the right wrapper by model ID.

**`tasks/dist.py`** — Extended `get_end_str()` to support additional model families: Zamba, Falcon, Mamba, MPT, and Bloomz, in addition to the original Gemma, LLaMA, and Qwen support.

**`run_layer_experiments.py`** — Model loading extended with architecture-specific configurations:
- 8-bit quantization path for Falcon-7B-Instruct and similar models (`load_in_8bit=True`, `torch_dtype=float16`).
- Zamba/Mamba-specific `torch_dtype=float16` override.
- A robust `get_model_input_device()` utility for resolving safe input devices when using `device_map="auto"` with offloaded models.

---

### Layer-Sweep Experiment Runner

**`run_layer_experiments.py`** — A new end-to-end script for sweeping residual stream patching experiments across all (or a range of) layers in one command:
- `--start-layer` / `--end-layer` flags for running a subset of layers (used for parallelizing across SLURM array jobs).
- Per-layer output directories with a `results.csv` and `patch_effect.png` saved automatically.
- `LocalPipeline` — a thin pipeline wrapper accepting strings, lists, or raw-input dicts, compatible with `FilterExperiment` batching without requiring a full `LMPipeline` instance.
- Integrated with `plot_patch_effect()` for immediate visualization at each layer.
- Debug-level token logging: prints the exact base vs. counterfactual token at the patched position for each sample, flagging when they are identical (which would make the intervention a no-op).

---

### Extended Counterfactual Templates

**`training.py`** — Several new counterfactual generation templates added to the original set:

| Template | Description |
|---|---|
| `ppkn_simpler_counterfactual_template_keep_payload_change_key` | Swaps only the query key across instances, keeping the payload value fixed. |
| `ppkn_simpler_counterfactual_template_split_key_loc_new_payload` | Splits key and location changes, then replaces the payload with a novel value not present in the original context. |
| `ppkn_simpler_counterfactual_template_split_key_loc_maybe` | A relaxed variant where swap indices are drawn from the full instance range (including the query index), allowing partial overlaps. |
| `ppkn_simpler_counterfactual_template_split_key_loc_change_up_prev_index` | Adds an additional shuffle of the (cf_query_index − 1) position to probe sensitivity to adjacent-context changes. |
| `my_ppkn_counterfactual_template_joint_pos_key` | A variant where the key at the counterfactual query index in the base input is also updated, aligning key and position signals. |
| `ppkn_simpler_counterfactual_template_split_key_loc_new_keys_and_payloads` | Replaces all keys and payloads with entirely new values not appearing in the original prompt. |
| `counterfactual_template_just_change_question_index` | Changes only the question index (new key + new payload for the queried position), leaving all other instances unchanged. |
| `tuple_binding_counterfactual_template_for_key` | Swaps the key of two instances, used for isolating pure key-binding signal. |
| `tuple_binding_counterfactual_template_for_pos` | Swaps both key and payload of two instances, isolating positional signal. |
| `tuple_binding_counterfactual_template_for_key_new_key` | Replaces the key and payload of the queried instance with entirely novel values. |

Also added:
- `sample_pkn_question_template` — An alternative question sampler that does not set ordinal variables, used with schemas that don't require positional ordinals.
- `get_counterfactual_datasets_mixed` and `sample_answerable_question_template_mixed` — Mixed-schema dataset generation with per-schema overrides for query categories and answer categories, enabling experiments where different schemas use different query structures in the same training run.

---

### Multi-Order and Multi-Schema Causal Models

**`grammar/task_to_causal_model.py`** — Additional causal model constructors:
- `multi_order_multi_schema_task_to_lookbacks_generic_causal_model_with_special_vars` — Extends the generic multi-order model with two derived binary variables: `isFirstTwo` (whether the answer pointer is one of the first two instances) and `isLast` (whether it is the final instance). Useful for probing coarser positional structure.
- `multi_order_multi_schema_task_to_lookbacks_generic_causal_model_with_pdfs` — Identical structure to the base multi-order model but accepts a `pdfs` argument for non-uniform sampling distributions over instances.
- `multi_order_multi_schema_task_to_lookbacks_keyload_causal_model` — A causal model variant that explicitly represents a `Key` variable for each instance (derived from the non-answer categories), and a `QueryKey` variable for the queried instance. This separates the key-retrieval computation from the positional-lookup computation at the causal level.

---

### Filler Sentences for Distractor Contexts

**`grammar/task_to_causal_model.py`** — The `filler_sentences` list has been greatly expanded (now over 600 entries), providing a richer pool of semantically neutral, entity-free distractor sentences for injecting between binding statements. This supports the `fillers=True` / `num_fillers_per_item` options in dataset generation, which test whether models are robust to irrelevant intervening text.

---

### SLURM Job Infrastructure

**`slurms_jobs/`** — Job scripts for running layer-sweep experiments on a SLURM cluster, including:
- `falcH1small.slurm` — Example array job that parallelizes across layers, one GPU per layer, for the Falcon-H1-3B-Instruct model on the SCHEMA_BOXES task.
- Memory management flags (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`) for preventing fragmentation OOM on smaller GPUs.

---

### Plotting

**`plotting.py`** — The `plot_patch_effect` function has been extended:
- `include_reflexive` flag for showing the reflexive (payload) category as a distinct band.
- `include_no_effect` and `include_invalid` flags for additional effect categories.
- `reflexive_label` and `lexical_label` parameters for customizing legend entries.
- The patch effect categories used in `run_layer_experiments.py` have been renamed from the original (`keyload` → `lexical`, `payload` → `reflexive`) to better reflect their mechanistic interpretation.

---

### Tokenizer-Agnostic Box-Label Extraction

**`run_layer_experiments.py`** and **`tasks/dist.py`** — A shared `_extract_boxes_answer_positions_from_offsets()` utility that uses character-offset mapping (rather than string matching) to locate numeric box labels in the tokenized prompt. This handles tokenizers that split multi-digit numbers across multiple tokens (e.g., "69" → ["6", "9"]), which caused silent failures in the original implementation.

---

## Original Files

The following files are unchanged from or closely follow the original paper's release:

- `CausalAbstraction/` — Copy of the [CausalAbstraction](https://github.com/atticusg/CausalAbstraction) codebase with minor quality-of-life tweaks.
- `grammar/schemas.py` — All binding task schema definitions.
- `grammar/grammar.py` — Core `Schema`, `BindingTask`, and `TaskFactory` classes.
- `example.ipynb` — Example notebook demonstrating the main interchange intervention.

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

---

For questions about the extensions, feel free to open an issue. For questions about the original work, see the [interactive blog post](https://yoav.ml/blog/2025/mixing-mechs/) or contact the original authors.
