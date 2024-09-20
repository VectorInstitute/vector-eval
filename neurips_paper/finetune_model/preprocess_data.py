from __future__ import annotations

import argparse
import os
from argparse import Namespace
from copy import deepcopy
from typing import Any
import numpy as np

import datasets
from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from vectorlm.utils.data_utils import Config

from collections import Counter
import random


def parse_args() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/config.yaml")
    return parser.parse_args()


def validate_config(config: Config) -> None:
    """Validate config arguments.

    Args:
    ----
        config: The main config.
    """
    preprocess_args = config.dataset_preprocess
    if preprocess_args.get("from_disk"):
        if "load_path" not in preprocess_args:
            msg = "`from_disk` set but `load_path` missing."
            raise KeyError(msg)
        path_exists = os.path.exists(preprocess_args.get("from_disk"))
        if not path_exists:
            msg = "`load_path` does not exist."
            raise Exception(msg)

    if preprocess_args.get("truncate") and preprocess_args.get("packing_type"):
        msg = "`truncate` and `packing_type` cannot both be set."
        raise ValueError(msg)

    if (not preprocess_args.get("truncate")) and (
        not preprocess_args.get("packing_type")
    ):
        print(
            "Warning: neither `truncate` nor `packing_type` are set. This ",
            "can cause issues during the forward pass of the model if ",
            "tokenized example lengths exceed max sequence lengths.",
        )

    if not preprocess_args.get("save_path"):
        msg = "`save_path` missing from config."
        raise KeyError(msg)


def tokenize_dataset(
    examples: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    data_field: str,
    pre_pend: str | None = None,
    truncate: bool = False,
    separator: str | None = None,
    add_bos_eos: bool = True,
) -> dict[str, list[int]]:
    """Tokenize the text dataset.

    Args:
    ----
        examples: The dictionary containing raw examples.
        tokenizer: The tokenizer.
        data_field: The dictionary key containing the examples to be tokenized.
        pre_pend: An optional prompt you can prepend.
        truncate: Whether to truncate examples that exceed context length.
        separator: A string separator s.t. everything before the separator
            is not going to be backpropped, and everything after will. Similar
            to supervised finetuning.
        add_bos_eos: Whether to add the BOS and EOS tokens to the tokenized
            sequence.

    Returns:
    -------
        A dictionary containing the input ids, labels, and attention masks.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    # Adding bos/eos
    if add_bos_eos:
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
    else:
        bos, eos = "", ""
    for example in examples[data_field]:
        # If we want to include a prepended prompt to each datapoint
        if pre_pend:
            prompt = f"{bos}{pre_pend}{example}{eos}"
        else:
            prompt = f"{bos}{example}{eos}"
        # If we've specified a separator present in each sequence
        if not separator:
            tokenized = tokenizer.encode(prompt, add_special_tokens=False)
            if truncate and len(tokenized) > tokenizer.model_max_length:
                tokenized = tokenized[: tokenizer.model_max_length - 1]
                tokenized.append(tokenizer.eos_token_id)
            all_labels.append(deepcopy(tokenized))
        else:
            if separator not in prompt:
                continue
            # Perform tokenization separately to allow for conditional prompting
            separation_idx = prompt.find(separator) + len(separator)
            prefix, postfix = prompt[:separation_idx], prompt[separation_idx:]
            tokenized_prefix = tokenizer.encode(
                prefix,
                add_special_tokens=False,
            )
            tokenized_postfix = tokenizer.encode(
                postfix,
                add_special_tokens=False,
            )
            tokenized = tokenized_prefix + tokenized_postfix
            if truncate and len(tokenized) > tokenizer.model_max_length:
                tokenized = tokenized[: tokenizer.model_max_length - 1]
                tokenized.append(tokenizer.eos_token_id)
            # We need to address this separately, because labels need to
            # backprop on bos/eos tokens
            if add_bos_eos:
                label = (
                    [tokenizer.bos_token_id]
                    + ([-100] * (len(tokenized_prefix) - 1))
                    + deepcopy(tokenized_postfix)
                )
            else:
                label = [-100] * len(tokenized_prefix) + deepcopy(tokenized_postfix)
            # If truncated, labels should be the same.
            if truncate and len(label) > tokenizer.model_max_length:
                label = label[: tokenizer.model_max_length - 1]
                label.append(tokenizer.eos_token_id)
            all_labels.append(label)
        all_input_ids.append(tokenized)
        all_attention_mask.append([1] * len(tokenized))

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def pack_examples(
    examples: dict[str, list[int]],
    tokenizer: PreTrainedTokenizer,
    overlap: int = 0,
    packing_type: str = "full",
    add_bos_eos: bool = True,
) -> dict[str, list[int]]:
    """Pack the tokenized dataset.

    Args:
    ----
        examples: The dictionary containing tokenized results.
        tokenizer: The tokenizer.
        overlap: The amount of overlap between two examples while packing.
        packing_type: In `partial` packing, original individual training
            examples are chunked with the option of `overlap`. In `full`
            packing, the entire dataset is fully packed meaning that no
            empty space is left in sequences.
        add_bos_eos: Whether to add the BOS and EOS tokens to the tokenized
            and packed sequence.

    Returns:
    -------
        Same type of input dictionary, but now with examples packed.
    """
    chunk_size = tokenizer.model_max_length
    if add_bos_eos:
        # For BOS and EOS tokens.
        chunk_size -= 2
        bos, eos = [tokenizer.bos_token_id], [tokenizer.eos_token_id]
    else:
        bos, eos = [], []
    stride = chunk_size - overlap
    all_keys = list(examples.keys())
    if packing_type == "full":
        joined_examples = {k: sum(examples[k], []) for k in all_keys}
        total_length = len(joined_examples["input_ids"])
        result = {}
        for k, v in joined_examples.items():
            value_chunked_lst = []
            for i in range(0, total_length, stride):
                if k != "attention_mask":
                    value_chunked_lst.append(bos + v[i : i + chunk_size] + eos)
                else:
                    if add_bos_eos:
                        # Need to do this explicitly because attention mask
                        # is just 1s or 0s.
                        value_chunked_lst.append([1] + v[i : i + chunk_size] + [1])
                    else:
                        value_chunked_lst.append(v[i : i + chunk_size])
    elif packing_type == "partial":
        result = {k: [] for k in examples}
        _key = all_keys[0]
        for idx in range(len(examples[_key])):
            total_length = len(examples[_key][idx])
            for key in all_keys:
                for i in range(0, total_length, stride):
                    if key != "attention_mask":
                        sliced_example = [
                            bos + examples[key][idx][i : i + chunk_size] + eos
                        ]
                    else:
                        if add_bos_eos:
                            sliced_example = [
                                [1] + examples[key][idx][i : i + chunk_size] + [1]
                            ]
                        else:
                            sliced_example = [examples[key][idx][i : i + chunk_size]]
                    result[key].extend(sliced_example)
    else:
        msg = "`packing_type` needs to either be `full` or `partial`."
        raise ValueError(msg)
    return result


def add_indices(
    dataset: datasets.Dataset,
    base_idx: int = 0,
) -> datasets.Dataset:
    """Add the `id` column to examples. This is used for dataset checkpointing.

    Args:
    ----
        dataset: The dataset we are adding the column to.
        base_idx: An optional base index to start the ids from. Default is 0.

    Returns:
    -------
        The dataset with the new `id` column.
    """
    indices = [i + base_idx for i in range(len(dataset))]
    return dataset.add_column("id", indices)


def apply_template(examples: dict[str, Any], prompt_template: str) -> dict[str, Any]:
    """Apply the prompt template to the dataset.

    Args:
    ----
        examples: The dataset examples.

    Returns:
    -------
        The dataset with the template applied.
    """
    final_text = []
    for input, output in zip(examples["input"], examples["output"]):
        final_text.append(
            prompt_template.format(
                instruction=input,
                response=output,
            )
        )
    return {"text": final_text}


def custom_train_test_split(ds, id_col, test_size=0.1, shuffle=True, seed=37):
    random.seed(seed)
    id_count = [elm[id_col] for elm in ds]
    id_count = dict(Counter(id_count))
    all_ids = list(id_count.keys())
    test_count = np.ceil(test_size * len(ds))
    test_ids = []
    curr_count = 0
    while curr_count < test_count:
        curr_id = random.choice(list(id_count.keys()))
        test_ids.append(curr_id)
        curr_count += id_count[curr_id]
        _ = id_count.pop(curr_id)

    train_ids = list(set(all_ids) - set(test_ids))
    assert len(set.intersection(set(train_ids), set(test_ids))) == 0

    return {
        "train": ds.filter(lambda x: x[id_col] in train_ids),
        "test": ds.filter(lambda x: x[id_col] in test_ids),
    }


def main(config: Config) -> None:
    """Definition of main function.

    Args:
    ----
        config: The main config.
    """
    preprocess_args = config.dataset_preprocess
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.model_max_length = config.train_parameters.max_seq_len
    if preprocess_args.from_disk:
        ds = datasets.load_from_disk(preprocess_args.load_path)
        if isinstance(ds, DatasetDict):
            if not preprocess_args.get("split"):
                msg = "Loaded dataset is a dictionary. Please specify `split`."
                raise KeyError(msg)
            ds = ds[preprocess_args.get("split")]
    else:
        ds = datasets.load_dataset(
            preprocess_args.load_path,
            split=preprocess_args.get("split"),
            name=preprocess_args.get("subset", None),
        )

    print("data loaded.")
    print(f"# examples: {len(ds)}\n")

    # using alpaca template from here:
    # https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=vITh0KVJ10qX
    prompt_template = {
        "template": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}""",
        "separator": "### Response:\n",
    }
    # apply prompt template to the dataset
    ds = ds.map(
        lambda examples: apply_template(examples, prompt_template["template"]),
        batched=True,
        batch_size=250,
        num_proc=32,
    )

    print("prompt template applied.")
    print(f"# examples: {len(ds)}\n")

    # train-val split
    TEST_SIZE = 0.1
    # ds = ds.train_test_split(test_size=TEST_SIZE, shuffle=True, seed=37)
    ds = custom_train_test_split(
        ds, id_col="source_id", test_size=TEST_SIZE, shuffle=True, seed=37
    )

    print(f"data split into train and val set ({TEST_SIZE} test).")
    print(f"# examples (train): {len(ds['train'])}")
    print(f"# examples (val): {len(ds['test'])}\n")

    for subset, ds in ds.items():
        subset = "val" if subset == "test" else subset

        will_pack = False
        if preprocess_args.get("packing_type"):
            will_pack = True

        if preprocess_args.get("add_bos_eos_tokens", True):
            special_tokens_created = isinstance(
                tokenizer.bos_token_id,
                int,
            ) and isinstance(
                tokenizer.eos_token_id,
                int,
            )
            if not special_tokens_created:
                msg = (
                    "BOS and EOS tokens are not set in the tokenizer.",
                    "Cannot add these tokens during tokenization.",
                )
                raise TypeError(msg)

        ds = ds.map(
            lambda examples: tokenize_dataset(
                examples,
                tokenizer,
                preprocess_args.data_field,
                preprocess_args.get("pre_pend"),
                preprocess_args.get("truncate", False),
                # preprocess_args.get("seperator"),
                prompt_template["separator"],  # this changes for non-instruct data
                # Note that if packing, we add the special tokens after packing.
                not will_pack and preprocess_args.get("add_bos_eos_tokens", True),
            ),
            batched=True,
            batch_size=250,
            remove_columns=ds.column_names,
            num_proc=32,
        )

        print(f"{subset} data tokenized.")
        input_ids = ds["input_ids"]
        print(f"# examples: {len(input_ids)}")
        num_tokens = [len(elm) for elm in input_ids]
        print(f"# tokens (mean): {np.mean(num_tokens)}")
        print(f"# tokens (total): {np.sum(num_tokens)}\n")

        if preprocess_args.get("packing_type"):
            ds = ds.map(
                lambda examples: pack_examples(
                    examples,
                    tokenizer,
                    preprocess_args.get("overlap", 0),
                    preprocess_args.packing_type,
                    will_pack and preprocess_args.get("add_bos_eos_tokens", True),
                ),
                batched=True,
                batch_size=2000,
                remove_columns=ds.column_names,
                num_proc=8,
            )

            print(f"{subset} data packed.")
            input_ids = ds["input_ids"]
            print(f"# examples: {len(input_ids)}")
            num_tokens = [len(elm) for elm in input_ids]
            print(f"# tokens (mean): {np.mean(num_tokens)}")
            print(f"# tokens (total): {np.sum(num_tokens)}\n")

        ds = add_indices(ds)
        ds.save_to_disk(os.path.join(preprocess_args.save_path, subset), max_shard_size="1GB")


if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config_path)
    validate_config(config)
    main(config)
