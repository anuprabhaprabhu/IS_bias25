import torch
import torchaudio
import pandas as pd
import numpy as np
# from datasets import load_metric
import evaluate 
from datasets import Dataset, Audio, Value
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
import re, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
from transformers import TrainingArguments
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

test_data_dir = '/home2/meenakshi.sirigiraju/feb13_ft/data_files/test'
healthy_data_dir='/home2/meenakshi.sirigiraju/feb13_ft/data_files/test_cate/healthy'
low_data_dir='/home2/meenakshi.sirigiraju/feb13_ft/data_files/test_cate/low'
mid_data_dir = '/home2/meenakshi.sirigiraju/feb13_ft/data_files/test_cate/mid'
high_data_dir = '/home2/meenakshi.sirigiraju/feb13_ft/data_files/test_cate/high'

raw_dataset = DatasetDict()
# raw_dataset["train"] = load_custom_dataset('train')
# raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate)

def custom_dataprep(data_dir):
    scp_entries = open(f"{data_dir}/audio_paths", 'r').readlines()
    txt_entries = open(f"{data_dir}/text", 'r').readlines()

    if len(scp_entries) == len(txt_entries):
        audio_dataset = Dataset.from_dict({"audio": [audio_path.split()[1].strip() for audio_path in scp_entries],
                        "sentence": [' '.join(text_line.split()[1:]).strip() for text_line in txt_entries]})

        audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))
        audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
        return(audio_dataset)

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch
def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

raw_dataset["test"] = custom_dataprep(test_data_dir)
raw_dataset["healthy"] = custom_dataprep(healthy_data_dir)
raw_dataset["low"] = custom_dataprep(low_data_dir)
raw_dataset["mid"] = custom_dataprep(mid_data_dir)
raw_dataset["high"] = custom_dataprep(high_data_dir)

print(raw_dataset)

# Modify the chars_to_remove_regex pattern to include the additional symbols
chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\(\)\*\=\_`\[\]\/\*°ː’•…]'
raw_dataset["test"] = raw_dataset["healthy"].map(remove_special_characters)
raw_dataset["healthy"] = raw_dataset["healthy"].map(remove_special_characters)
raw_dataset["low"] = raw_dataset["low"].map(remove_special_characters)
raw_dataset["mid"] = raw_dataset["mid"].map(remove_special_characters)
raw_dataset["high"] = raw_dataset["high"].map(remove_special_characters)

vocab_healthy = raw_dataset["test"].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=raw_dataset["test"].column_names)
vocab_healthy = raw_dataset["healthy"].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=raw_dataset["healthy"].column_names)
vocab_low = raw_dataset["low"].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=raw_dataset["low"].column_names)
vocab_mid = raw_dataset["mid"].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=raw_dataset["mid"].column_names)
vocab_high = raw_dataset["high"].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=raw_dataset["high"].column_names)


from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

raw_dataset["test"] = raw_dataset["test"].cast_column("audio", Audio(sampling_rate=16000))
raw_dataset["test"] = raw_dataset["test"].map(prepare_dataset, num_proc=2)
raw_dataset["healthy"] = raw_dataset["healthy"].cast_column("audio", Audio(sampling_rate=16000))
raw_dataset["healthy"] = raw_dataset["healthy"].map(prepare_dataset, num_proc=2)
raw_dataset["low"] = raw_dataset["low"].cast_column("audio", Audio(sampling_rate=16000))
raw_dataset["low"] = raw_dataset["low"].map(prepare_dataset, num_proc=2)
raw_dataset["mid"] = raw_dataset["mid"].cast_column("audio", Audio(sampling_rate=16000))
raw_dataset["mid"] = raw_dataset["mid"].map(prepare_dataset, num_proc=2)
raw_dataset["high"] = raw_dataset["high"].cast_column("audio", Audio(sampling_rate=16000))
raw_dataset["high"] = raw_dataset["high"].map(prepare_dataset, num_proc=2)

# Concatenate cv_swahili_train and cv_swahili_validate
# combined_train_validate = concatenate_datasets([raw_dataset["train"], raw_dataset["val"]])

class DataCollatorCTCWithPadding():
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __init__(self, processor: Wav2Vec2Processor, padding: Union[bool, str] = True):
        self.processor = processor  # Initialize processor as an attribute
        self.padding = padding
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
# wer_metric = load_metric("wer")
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}



print("Evaluate on test  ")


processor = Wav2Vec2Processor.from_pretrained("/home2/meenakshi.sirigiraju/data/hugg_models/save_model")
model = Wav2Vec2ForCTC.from_pretrained("/home2/meenakshi.sirigiraju/data/hugg_models/save_model").cuda()

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  return batch

results = raw_dataset["test"].map(map_to_result)
print("-> Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

results = raw_dataset["healthy"].map(map_to_result)
print("For healthy -> Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

results = raw_dataset["low"].map(map_to_result)
print("For low -> Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

results = raw_dataset["mid"].map(map_to_result)
print("For mid -> Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

results = raw_dataset["high"].map(map_to_result)
print("For high -> Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

print("Completted !!!!")