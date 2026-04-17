import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset

_DTYPE_MAP = {"int32": np.int32, "int64": np.int64, "float32": np.float32}


def pre_processing_chat(conversations, add_system_ratio=0.2, system_prompts=None):
    # tool use 数据完整保留不做处理
    if any(conv.get("tools") for conv in conversations):
        return conversations

    if system_prompts is None:
        system_prompts = []
    # 概率性添加system
    if system_prompts and conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{"role": "system", "content": random.choice(system_prompts)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if "<think>\n\n</think>\n\n" in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


class PretrainDataset(Dataset):
    def __init__(
        self,
        data_path=None,
        tokenizer=None,
        max_length=512,
        samples=None,
        sample_indices=None,
        tokens_file="tokens.bin",
        index_file="index.bin",
        metadata_file="metadata.json",
        tokens_dtype_name="int32",
        index_dtype_name="int64",
        dataset_version=1,
    ):
        super().__init__()
        if data_path is None:
            raise ValueError("data_path is required")
        if tokenizer is not None:
            raise ValueError("PretrainDataset now reads pretokenized mmap data; tokenizer is no longer accepted")
        if samples is not None:
            raise ValueError("PretrainDataset now reads pretokenized mmap data; samples is no longer accepted")
        self.max_length = max_length
        self.data_path = Path(data_path)
        self._tokens_file = tokens_file
        self._index_file = index_file
        self._tokens_dtype = _DTYPE_MAP[tokens_dtype_name]
        self._index_dtype = _DTYPE_MAP[index_dtype_name]
        self.metadata = load_pretokenized_metadata(
            self.data_path, metadata_file=metadata_file, dataset_version=dataset_version
        )
        self.pad_token_id = int(self.metadata["pad_token_id"])
        self.sample_count = int(self.metadata["sample_count"])
        self.sample_indices = None if sample_indices is None else np.asarray(sample_indices, dtype=np.int64)
        self._tokens = None
        self._index = None

    def __len__(self):
        return int(self.sample_count if self.sample_indices is None else len(self.sample_indices))

    def sample_lengths(self):
        self._ensure_memmaps()
        if self.sample_indices is None:
            return np.asarray(self._index[:, 1], dtype=np.int64)
        return np.asarray(self._index[self.sample_indices, 1], dtype=np.int64)

    def __getitem__(self, index):
        self._ensure_memmaps()
        sample_index = int(index if self.sample_indices is None else self.sample_indices[index])
        token_start, token_length = self._index[sample_index]
        tokens = self._tokens[int(token_start) : int(token_start + token_length)]
        return torch.tensor(tokens, dtype=torch.long)

    def _ensure_memmaps(self):
        if self._tokens is None:
            self._tokens = np.memmap(
                self.data_path / self._tokens_file,
                dtype=self._tokens_dtype,
                mode="r",
                shape=(int(self.metadata["token_count"]),),
            )
        if self._index is None:
            self._index = np.memmap(
                self.data_path / self._index_file,
                dtype=self._index_dtype,
                mode="r",
                shape=(self.sample_count, 2),
            )


def get_attention_mask_for_packed_sequence(x, eos_token_id, pad_token_id):
    """Build a packed causal mask and reset position IDs for one fixed-length row."""

    T = x.size(0)
    valid = x.ne(pad_token_id)
    valid_count = int(valid.sum().item())

    attention_mask = torch.zeros(T, T, dtype=torch.bool, device=x.device)
    position_ids = torch.zeros(T, dtype=torch.long, device=x.device)
    if valid_count == 0:
        attention_mask.fill_diagonal_(True)
        return attention_mask, position_ids

    eos_indices = ((x == eos_token_id) & valid).nonzero(as_tuple=True)[0]
    last_valid_idx = valid.nonzero(as_tuple=True)[0][-1]
    if eos_indices.numel() == 0 or int(eos_indices[-1].item()) != int(last_valid_idx.item()):
        eos_indices = torch.cat([eos_indices, last_valid_idx.view(1)])

    reps = torch.cat([eos_indices[:1] + 1, eos_indices[1:] - eos_indices[:-1]])
    seq_starts = torch.cat([eos_indices.new_zeros(1), eos_indices[:-1] + 1])
    token_starts = torch.repeat_interleave(seq_starts, reps)
    repeated_idx = torch.full((T,), int(last_valid_idx.item()), dtype=torch.long, device=x.device)
    repeated_idx[:valid_count] = torch.repeat_interleave(eos_indices, reps)

    valid_positions = torch.arange(valid_count, device=x.device, dtype=torch.long)
    position_ids[:valid_count] = valid_positions - token_starts

    row_idx = torch.arange(T, device=x.device, dtype=torch.long).view(-1, 1).expand(-1, T)
    col_idx = torch.arange(T, device=x.device, dtype=torch.long).view(1, -1).expand(T, -1)
    attention_mask = (row_idx >= col_idx) & (row_idx <= repeated_idx.view(1, -1))
    attention_mask &= valid.view(-1, 1) & valid.view(1, -1)
    pad_indices = (~valid).nonzero(as_tuple=True)[0]
    if pad_indices.numel() > 0:
        # Keep padded queries from becoming fully masked rows. Older SDPA kernels
        # can turn all-masked padded queries into NaNs even though labels ignore them.
        attention_mask[pad_indices, pad_indices] = True
    return attention_mask, position_ids


class PretrainDataCollator:
    """Pack raw samples into fixed rows, then build packed masks and reset position IDs."""

    def __init__(self, eos_token_id=2, pad_token_id=0, max_seq_len=512):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def _extract_input_ids(self, feature):
        if torch.is_tensor(feature):
            return feature
        if isinstance(feature, (tuple, list)) and feature and torch.is_tensor(feature[0]):
            return feature[0]
        raise TypeError(f"Unsupported pretrain feature type: {type(feature)!r}")

    def _finalize_row(self, samples):
        row = torch.full((self.max_seq_len,), self.pad_token_id, dtype=samples[0].dtype)
        offset = 0
        for sample in samples:
            next_offset = offset + int(sample.numel())
            row[offset:next_offset] = sample
            offset = next_offset
        return row

    def __call__(self, features):
        packed_rows = []
        current_row = []
        current_length = 0

        for feature in features:
            input_ids = self._extract_input_ids(feature)
            sample_length = int(input_ids.numel())
            if sample_length > self.max_seq_len:
                print(
                    f"[pretrain-collator] truncating sample from length {sample_length} to {self.max_seq_len}",
                    flush=True,
                )
                # Force a terminal EOS token so packed rows do not merge a truncated
                # sample with the next document boundary.
                input_ids = torch.cat(
                    [
                        input_ids[: self.max_seq_len - 1],
                        input_ids.new_tensor([self.eos_token_id]),
                    ]
                )
                sample_length = self.max_seq_len
            if current_row and current_length + sample_length > self.max_seq_len:
                packed_rows.append(self._finalize_row(current_row))
                current_row = []
                current_length = 0

            current_row.append(input_ids)
            current_length += sample_length

        if current_row:
            packed_rows.append(self._finalize_row(current_row))

        input_ids = torch.stack(packed_rows)
        labels = input_ids.clone()
        labels[input_ids == self.pad_token_id] = -100
        attention_masks = []
        position_ids = []
        for row in input_ids:
            row_attention_mask, row_position_ids = get_attention_mask_for_packed_sequence(
                row,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
            attention_masks.append(row_attention_mask)
            position_ids.append(row_position_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": torch.stack(position_ids),
            "attention_mask": torch.stack(attention_masks),
        }


def pretokenized_dataset_exists(path, metadata_file="metadata.json"):
    return (Path(path) / metadata_file).is_file()


def load_pretokenized_metadata(
    path,
    metadata_file="metadata.json",
    dataset_version=1,
    index_file="index.bin",
    index_dtype_name="int64",
):
    metadata_path = Path(path) / metadata_file
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("version") != dataset_version:
            raise ValueError(
                f"Unsupported pretokenized dataset version {metadata.get('version')}; expected {dataset_version}"
            )
        return metadata
    # Derive metadata from binary files when metadata.json is missing
    index_path = Path(path) / index_file
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Neither {metadata_path} nor {index_path} found. "
            "Run the pretokenization pipeline before starting pretraining."
        )
    index_dtype = _DTYPE_MAP[index_dtype_name]
    index_data = np.memmap(str(index_path), dtype=index_dtype, mode="r")
    sample_count = len(index_data) // 2  # each entry is (offset, length)
    return {
        "version": dataset_version,
        "sample_count": sample_count,
        "pad_token_id": 0,
    }


def pretokenized_sample_count(path, metadata_file="metadata.json", dataset_version=1):
    return int(
        load_pretokenized_metadata(path, metadata_file=metadata_file, dataset_version=dataset_version)["sample_count"]
    )


def build_pretokenized_corpus(
    input_path,
    output_dir,
    tokenizer,
    max_length,
    *,
    overwrite=False,
    progress_interval=50000,
    tokens_file="tokens.bin",
    index_file="index.bin",
    metadata_file="metadata.json",
    tokens_dtype_name="int32",
    index_dtype_name="int64",
    dataset_version=1,
):
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    tokens_dtype = _DTYPE_MAP[tokens_dtype_name]
    index_dtype = _DTYPE_MAP[index_dtype_name]

    source_path = Path(input_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"{source_path} not found")

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass overwrite=True to rebuild it")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    tokens_tmp = temp_dir / tokens_file
    index_tmp = temp_dir / index_file

    token_count = 0
    sample_count = 0
    with (
        source_path.open(encoding="utf-8") as source,
        tokens_tmp.open("wb") as tokens_fp,
        index_tmp.open("wb") as index_fp,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "text" not in payload:
                raise ValueError(f"{source_path}:{line_number} is missing the 'text' field")
            token_ids = tokenizer(
                str(payload["text"]), add_special_tokens=False, max_length=max_length - 2, truncation=True
            ).input_ids
            token_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
            np.asarray(token_ids, dtype=tokens_dtype).tofile(tokens_fp)
            np.asarray((token_count, len(token_ids)), dtype=index_dtype).tofile(index_fp)
            token_count += len(token_ids)
            sample_count += 1
            if progress_interval > 0 and sample_count % progress_interval == 0:
                print(
                    f"[pretokenize] processed {sample_count} samples / {token_count} tokens",
                    flush=True,
                )

    metadata = {
        "version": dataset_version,
        "max_length": int(max_length),
        "sample_count": int(sample_count),
        "token_count": int(token_count),
        "tokens_dtype": tokens_dtype_name,
        "index_dtype": index_dtype_name,
        "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "source_path": str(source_path),
    }
    with (temp_dir / metadata_file).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(temp_dir, output_dir)
    return metadata


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        _require_hf_datasets()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features(
            {
                "conversations": [
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                        "reasoning_content": Value("string"),
                        "tools": Value("string"),
                        "tool_calls": Value("string"),
                    }
                ]
            }
        )
        self.samples = load_dataset("json", data_files=jsonl_path, split="train", features=features)
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        _require_hf_datasets()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]  # 是一个 list，里面包含若干 {role, content}
        rejected = sample["rejected"]  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding="max_length"
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding="max_length"
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        _require_hf_datasets()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1], tokenize=False, open_thinking=use_thinking, add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": ""}


class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample["conversations"])
        return {"messages": messages, "tools": tools, "gt": sample["gt"]}


def _require_hf_datasets():
    if load_dataset is None or Features is None or Value is None:
        raise ImportError("The 'datasets' package is required for supervised/chat dataset loading")


if __name__ == "__main__":
    pass
