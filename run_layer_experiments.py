#!/usr/bin/env python3
"""
Run patch effect experiments across all layers of a model.

This script iterates through all layers of a specified model, runs patch effect
experiments for each layer, and saves the results as plots (PNG/PDF) and CSV files.
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append("CausalAbstraction")

# Suppress transformers logging
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
import transformers

transformers.logging.set_verbosity_error()

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from training import (
    sample_answerable_question_template,
    get_counterfactual_datasets,
    ppkn_simpler_counterfactual_template_split_key_loc,
)
from grammar.task_to_causal_model import multi_order_multi_schema_task_to_lookbacks_generic_causal_model
from grammar.schemas import SCHEMA_BOXES
from tasks.dist import (
    get_end_str,
    format_prompt,
    to_str_tokens,

    run_with_cf_hf,
    try_schema_checker,
    _num_layers,
)
from plotting import plot_patch_effect


def get_schema_by_name(schema_name: str):
    """Get schema object by name."""
    from grammar.schemas import (
        SCHEMA_FILLING_LIQUIDS,
        SCHEMA_PEOPLE_AND_OBJECTS,
        SCHEMA_PROGRAMMING_PEOPLE_DICT,
        SCHEMA_MUSIC_PERFORMANCE,
        SCHEMA_LAB_EXPERIMENTS,
        SCHEMA_CHEMISTRY_EXPERIMENTS,
        SCHEMA_TRANSPORTATION,
        SCHEMA_SPORTS_EVENTS,
        SCHEMA_SPACE_OBSERVATIONS,
        SCHEMA_BOXES,
    )
    
    schemas = {
        "SCHEMA_FILLING_LIQUIDS": SCHEMA_FILLING_LIQUIDS,
        "SCHEMA_PEOPLE_AND_OBJECTS": SCHEMA_PEOPLE_AND_OBJECTS,
        "SCHEMA_PROGRAMMING_PEOPLE_DICT": SCHEMA_PROGRAMMING_PEOPLE_DICT,
        "SCHEMA_MUSIC_PERFORMANCE": SCHEMA_MUSIC_PERFORMANCE,
        "SCHEMA_LAB_EXPERIMENTS": SCHEMA_LAB_EXPERIMENTS,
        "SCHEMA_CHEMISTRY_EXPERIMENTS": SCHEMA_CHEMISTRY_EXPERIMENTS,
        "SCHEMA_TRANSPORTATION": SCHEMA_TRANSPORTATION,
        "SCHEMA_SPORTS_EVENTS": SCHEMA_SPORTS_EVENTS,
        "SCHEMA_SPACE_OBSERVATIONS": SCHEMA_SPACE_OBSERVATIONS,
        "SCHEMA_BOXES": SCHEMA_BOXES,
    }
    
    if schema_name not in schemas:
        raise ValueError(f"Unknown schema name: {schema_name}. Available: {list(schemas.keys())}")
    
    return schemas[schema_name]


def sanitize_model_name(model_id: str) -> str:
    """Convert model_id to a filesystem-safe folder name."""
    return model_id.replace("/", "_")

def get_model_input_device(model) -> torch.device:
    """
    Resolve a safe device for input tensors when model was loaded with device_map='auto'.
    Avoids choosing 'meta'/'disk' placements used by offloading.
    """
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        # Prefer CUDA if any module is placed there.
        for dev in model.hf_device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str):
                d = dev.lower()
                if d.startswith("cuda"):
                    return torch.device(dev)

        # Otherwise pick the first concrete non-meta/non-disk mapping.
        for dev in model.hf_device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str):
                d = dev.lower()
                if d not in {"meta", "disk"}:
                    return torch.device(dev)

    # Fallback: first non-meta parameter/buffer device.
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    for b in model.buffers():
        if b.device.type != "meta":
            return b.device

    # Last resort for fully offloaded edge cases.
    return torch.device("cpu")


def _extract_boxes_answer_positions_from_offsets(prompt: str, tokenizer, metadata: dict, num_instances: int):
    """
    Tokenizer-agnostic extraction for SCHEMA_BOXES numeric answers.

    Uses character offsets to map each "Box <number>" mention to the token index
    containing that number's first digit, which works even when numbers are split
    across multiple tokens (e.g., "6" + "9" for 69).
    """
    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer did not return offset mappings.")

    # Fast tokenizers may return either:
    # - unbatched: List[Tuple[start, end]]
    # - batched:   List[List[Tuple[start, end]]]
    if offsets and isinstance(offsets[0], (list, tuple)):
        first = offsets[0]
        if isinstance(first, tuple) and len(first) == 2:
            # Already unbatched list of (start, end).
            pass
        elif isinstance(first, list):
            # Batched output for a single prompt.
            offsets = first

    answer_indices = []
    answer_labels = []
    for match in re.finditer(r"Box\s*(\d+)", prompt):
        label = match.group(1)
        digit_start = match.start(1)

        token_idx = None
        for idx, (start, end) in enumerate(offsets):
            # Special tokens often map to (0, 0); skip them.
            if start == end:
                continue
            if start <= digit_start < end:
                token_idx = idx
                break

        if token_idx is not None:
            answer_indices.append(token_idx)
            answer_labels.append(label)

    # Only context statements should have a numeric "Box <n>" pattern.
    if len(answer_indices) != num_instances:
        raise AssertionError(
            f"Offset-based extraction expected {num_instances} indices, got {len(answer_indices)}."
        )

    def _as_label(x):
        m = re.search(r"\d+", str(x))
        return m.group(0) if m else None

    keyload_label = _as_label(metadata["keyload"])
    payload_label = _as_label(metadata["payload"])
    keyload_index = answer_labels.index(keyload_label) if keyload_label in answer_labels else None
    payload_index = answer_labels.index(payload_label) if payload_label in answer_labels else None

    return answer_indices, answer_labels, keyload_index, payload_index

def run_experiment_for_layer(
    model,
    tokenizer,
    train_ds,
    schema,
    num_instances: int,
    num_samples: int,
    layer: int,
    cat_to_query: int,
    model_id: str,
    generate: bool = False,
):
    """
    Run the patch effect experiment for a single layer.
    
    Returns:
        pd.DataFrame: Results dataframe
    """
    results = {
        "normal": [],
        "cf": [],
        "source_pos": [],
        "positional_index": [],
        "keyload_index": [],
        "payload_index": [],
        "layer": [],
        "prediction": [],
        "positional_prediction": [],
        "payload_prediction": [],
        "keyload_prediction": [],
        "no_effect_prediction": [],
        "patch_effect": [],
        "dist": [],
        "distance": [],
        "generated": [],
    }
    
    train = train_ds[schema.name][schema.name]
    token_positions = [-1]
    base_indices = [i for i, t in enumerate(prompt_str_tokenized) if schema.matchers[cat_to_query](t.strip())]
    cf_indices = [i for i, t in enumerate(to_str_tokens(tokenizer, cf_prompt)) if schema.matchers[cat_to_query](t.strip())]
    base_pos, source_pos = [base_indices[pos_index]], [cf_indices[pos_index]]
    end_str = get_end_str(model_id)
    model_id_str = model_id

    def _extract_numeric_label(token: str):
        match = re.search(r"\d+", str(token))
        return match.group(0) if match else None

    for cur_index in tqdm(range(num_samples), desc=f"Layer {layer}"):
        prompt = format_prompt(tokenizer, train[cur_index]["input"]["raw_input"])
        cf_prompt = format_prompt(tokenizer, train[cur_index]["counterfactual_inputs"][0]["raw_input"])
        prompt_str_tokenized = to_str_tokens(tokenizer, prompt)
        metadata = train[cur_index]["input"]["metadata"]

        answer_indices = []
        answer_labels = []
        keyload_index = None
        payload_index = None
        for i, token in enumerate(prompt_str_tokenized):
            if "qwen" in model_id_str.lower() and i < 10:
                continue
            

            if schema.matchers[cat_to_query](token.strip()):
                answer_indices.append(i)
                answer_labels.append(_extract_numeric_label(token) if schema.name == "boxes" and cat_to_query == 1 else token)
                if prompt_str_tokenized[i].lower().strip() in metadata["keyload"].lower().strip():
                    keyload_index = len(answer_indices) - 1

                if prompt_str_tokenized[i].lower().strip() in metadata["payload"].lower().strip():
                    payload_index = len(answer_indices) - 1

        if (
            schema.name == "boxes"
            and cat_to_query == 1
            and (
                len(answer_indices) != num_instances
                or keyload_index is None
                or payload_index is None
            )
        ):
            # Fallback for tokenizers that split numeric labels into multiple tokens.
            answer_indices, answer_labels, keyload_index, payload_index = _extract_boxes_answer_positions_from_offsets(
                prompt, tokenizer, metadata, num_instances
            )
            answer_labels = [_extract_numeric_label(prompt_str_tokenized[idx]) for idx in answer_indices]
        assert (
            len(answer_indices) == num_instances
        ), f"Expected {num_instances} answer indices, got {len(answer_indices)}.\nPrompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."
        assert (
            keyload_index is not None
        ), f"Keyload [{metadata['keyload']}] index is None. Prompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."

        assert (
            payload_index is not None
        ), f"Payload [{metadata['payload']}] index is None. Prompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."

        pos_index = metadata["dst_index"]
        token_positions = [answer_indices[pos_index]]
        base_pos = [answer_indices[pos_index]]
        source_pos = [cf_answer_indices[pos_index]]


        target_idx = token_positions[0]
        
        # חילוץ הטוקנים הטקסטואליים
        base_tokens = to_str_tokens(tokenizer, prompt)
        cf_tokens = to_str_tokens(tokenizer, cf_prompt)
        
        # זיהוי הערכים במיקום ההזרקה
        base_val = base_tokens[target_idx] if target_idx < len(base_tokens) else "OUT_OF_BOUNDS"
        cf_val = cf_tokens[target_idx] if target_idx < len(cf_tokens) else "OUT_OF_BOUNDS"
        
        # הדפסת הממצאים
        print(f"\n{'='*20} בדיקת דגימה {cur_index} {'='*20}")
        print(f"טוקן במקור (Base): '{base_val}'")
        print(f"טוקן להזרקה (Source): '{cf_val}'")
        
        # הדפסת הקשר (Context) של 3 טוקנים לפני ואחרי
        start_ctx = max(0, target_idx - 3)
        end_ctx = min(len(base_tokens), target_idx + 4)
        context = "".join(base_tokens[start_ctx:end_ctx])
        print(f"הקשר ב-Base: ...{context}...")
        
        if base_val == cf_val:
            print("⚠️  שים לב: הערכים זהים! ההתערבות לא תשנה דבר ב-Residual Stream.")
        print(f"{'='*50}\n")



        with run_with_cf_hf(
            model, tokenizer, prompt, cf_prompt, layer_idx=layer, token_positions=base_pos, alpha=1
        ):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            # START CHANGE
            with torch.no_grad():
                # IMPORTANT: This line must be indented further than the 'with' above it
                logits = model(input_ids).logits
            # END CHANGE

            # Get token IDs directly from input_ids instead of re-tokenizing strings
            token_ids_at_answer_positions = input_ids[0, answer_indices].tolist()
            values = logits[0, -1, token_ids_at_answer_positions]

            pos_pred = values.argmax().item()

            if generate:
                pred_ids = model.generate(input_ids, max_new_tokens=schema.max_new_tokens, do_sample=False)
                pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                pred = pred[pred.find(end_str) + len(end_str) :]
            else:
                if schema.name == "boxes" and cat_to_query == 1:
                    pred = answer_labels[pos_pred]
                else:
                    pred = prompt_str_tokenized[answer_indices[pos_pred]]
            
            # Strip whitespace and clean prediction to ensure proper matching
            # Remove all whitespace characters (spaces, tabs, newlines, etc.)
            pred = re.sub(r'\s+', '', pred.strip())

            if try_schema_checker(pred, metadata["positional"], schema):
                patch_effect = "positional"
            elif try_schema_checker(pred, metadata["keyload"], schema):
                patch_effect = "lexical"
            elif try_schema_checker(pred, metadata["payload"], schema):
                patch_effect = "reflexive"
            elif try_schema_checker(pred, metadata["no_effect"], schema):
                patch_effect = "no_effect"
            else:
                patch_effect = "mixed"

            results["normal"].append(prompt)
            results["cf"].append(cf_prompt)
            results["source_pos"].append(metadata["src_positional_index"])
            results["positional_index"].append(pos_index)
            results["keyload_index"].append(keyload_index)
            results["payload_index"].append(payload_index)
            results["layer"].append(layer)
            results["positional_prediction"].append(metadata["positional"])
            results["payload_prediction"].append(metadata["payload"])
            results["keyload_prediction"].append(metadata["keyload"])
            results["no_effect_prediction"].append(metadata["no_effect"])
            results["patch_effect"].append(patch_effect)
            results["prediction"].append(pred)
            results["dist"].append(values.tolist())
            results["distance"].append(pos_index - pos_pred)
            results["generated"].append(generate)

    df = pd.DataFrame(results)
    return df


class LocalPipeline:
    """Thin pipeline wrapper around an already-loaded model and tokenizer.

    Accepts str, list[str], dict{"raw_input": ...}, or list[dict] so that
    FilterExperiment can pass batches of dataset items directly.
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token
        )

    def _to_text_list(self, inputs):
        if isinstance(inputs, str):
            return [inputs]
        if isinstance(inputs, dict):
            return [inputs["raw_input"]]
        # list of dicts or list of strings
        return [x["raw_input"] if isinstance(x, dict) else x for x in inputs]

    def generate(self, inputs):
        texts = self._to_text_list(inputs)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in texts
            ]
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        self.tokenizer.padding_side = prev_side
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return out[:, enc["input_ids"].shape[1]:]

    def dump(self, output):
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


def save_results(df: pd.DataFrame, fig, output_dir: Path, layer: int):
    """Save plot (PNG) and CSV for a layer in a layer-specific subdirectory."""
    # Create layer-specific directory
    layer_dir = output_dir / f"layer_{layer}"
    layer_dir.mkdir(exist_ok=True)
    
    # Save CSV with descriptive name
    csv_path = layer_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    
    # Save PNG plot with descriptive name
    png_path = layer_dir / "patch_effect.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  Saved PNG: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run patch effect experiments across all layers of a model"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model ID from HuggingFace (default: google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=20,
        help="Number of instances (default: 20)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples (default: 50)",
    )
    parser.add_argument(
        "--cat-to-query",
        type=int,
        default=1,
        help="Category to query (default: 1)",
    )
    parser.add_argument(
        "--schema-name",
        type=str,
        default="SCHEMA_BOXES",
        help="Schema name (default: SCHEMA_BOXES)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Use generation instead of logits (default: False)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token for using restricted models (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: model name)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset generation (default: 42)",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=None,
        help="Start layer index (0-based, inclusive). If not specified, starts from layer 0.",
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=None,
        help="End layer index (0-based, inclusive). If not specified, goes to the last layer.",
    )
    
    args = parser.parse_args()
    
    # Get schema
    schema = get_schema_by_name(args.schema_name)
    print(f"[+] Using schema: {args.schema_name}")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name = sanitize_model_name(args.model_id)
        output_dir = Path(model_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"[+] Output directory: {output_dir.absolute()}")
    
   # ... inside main() ...

    print(f"[+] Loading model: {args.model_id}")

    # 1. Base Configuration
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,  # Default for modern models
    }

    # 2. Add Token if present
    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    # 3. Specific overrides for 8-bit models (e.g. falcon-7b-instruct)
    if "falcon-7b-instruct" in args.model_id.lower() or "8bit" in args.model_id.lower():
        print(f"[+] Applying 8-bit specific model loading configurations")
        model_kwargs.update({
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,  # 8-bit usually prefers fp16
        })
        
    elif "zamba" in args.model_id.lower():
        print(f"[+] Applying Zamba/Mamba specific model loading configurations")
        model_kwargs["torch_dtype"] = torch.float16

    # 4. Load Model
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # 5. Load Tokenizer
    tokenizer_kwargs = {"token": args.hf_token} if args.hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tokenizer_kwargs)
    tokenizer.padding_side = "left"

    # Get number of layers
    num_layers = _num_layers(model)
    print(f"[+] Model has {num_layers} layers")

    # Build pipeline for dataset filtering
    pipeline = LocalPipeline(model, tokenizer)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[+] Random seed set to: {args.seed}")

    # Setup causal model and datasets
    print(f"[+] Setting up causal model and datasets")
    causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
        [schema], args.num_instances, num_fillers_per_item=0, fillers=False
    )
    causal_models = {schema.name: causal_model}

    counterfactual_template = ppkn_simpler_counterfactual_template_split_key_loc

    train_ds, test_ds, fps = get_counterfactual_datasets(
        pipeline,
        [schema],
        num_samples=args.num_samples,
        num_instances=args.num_instances,
        cat_indices_to_query=[0],
        answer_cat_id=args.cat_to_query,
        do_assert=True,
        do_filter=True,
        counterfactual_template=counterfactual_template,
        causal_models=causal_models,
        sample_an_answerable_question=sample_answerable_question_template,
    )
    
    # Determine layer range
    start_layer = args.start_layer if args.start_layer is not None else 0
    end_layer = args.end_layer if args.end_layer is not None else (num_layers - 1)
    
    # Validate layer range
    if start_layer < 0:
        raise ValueError(f"start-layer must be >= 0, got {start_layer}")
    if end_layer >= num_layers:
        raise ValueError(f"end-layer must be < {num_layers}, got {end_layer}")
    if start_layer > end_layer:
        raise ValueError(f"start-layer ({start_layer}) must be <= end-layer ({end_layer})")
    
    layer_range = range(start_layer, end_layer + 1)
    num_layers_to_process = len(layer_range)
    
    # Iterate over specified layer range
    print(f"[+] Running experiments for layers {start_layer} to {end_layer} (inclusive, {num_layers_to_process} layers)...")
    for layer in tqdm(layer_range, desc="Layers"):
        print(f"\n[+] Processing layer {layer}/{num_layers - 1}")
        
        # Run experiment
        df = run_experiment_for_layer(
            model,
            tokenizer,
            train_ds,
            schema,
            args.num_instances,
            args.num_samples,
            layer,
            args.cat_to_query,
            args.model_id,
            generate=args.generate,
        )
        
        # Generate plot
        fig, ax = plot_patch_effect(df, include_reflexive=True, highest_near_pos=0)
        
        # Save results
        save_results(df, fig, output_dir, layer)
        
        # Close figure to free memory
        plt.close(fig)
    
    print(f"\n[+] All experiments completed! Results saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
