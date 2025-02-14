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


train_data_dir='/home2/meenakshi.sirigiraju/feb13_ft/data_files/one/train'
val_data_dir='/home2/meenakshi.sirigiraju/feb13_ft/data_files/one/val'
test_data_dir = '/home2/meenakshi.sirigiraju/feb13_ft/data_files/test'

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

raw_dataset["train"] = custom_dataprep(train_data_dir)
raw_dataset["val"] = custom_dataprep(val_data_dir)
raw_dataset["test"] = custom_dataprep(test_data_dir)
print(raw_dataset)

# Modify the chars_to_remove_regex pattern to include the additional symbols
chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\(\)\*\=\_`\[\]\/\*°ː’•…]'
raw_dataset["train"] = raw_dataset["train"].map(remove_special_characters)
raw_dataset["val"] = raw_dataset["val"].map(remove_special_characters)
raw_dataset["test"] = raw_dataset["test"].map(remove_special_characters)
vocab_train = raw_dataset["train"].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=raw_dataset["train"].column_names)
vocab_val = raw_dataset["val"].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=raw_dataset["val"].column_names)
vocab_test = raw_dataset["test"].map(extract_all_chars, batched=True, batch_size=-1, remove_columns=raw_dataset["test"].column_names)

# print(type(vocab_train))
# print(vocab_train)
# sys.exit()
# Convert "vocab" column from each dataset to sets and union them
vocab_set_train = set(vocab_train["vocab"][0])
vocab_set_val = set(vocab_val["vocab"][0])
vocab_set_test = set(vocab_test["vocab"][0])
vocab_set = vocab_set_train | vocab_set_val | vocab_set_test
vocab_list = list(vocab_set)
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict.pop('י', None)
print(vocab_dict)
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)


from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


raw_dataset["train"] = raw_dataset["train"].cast_column("audio", Audio(sampling_rate=16000))
raw_dataset["train"] = raw_dataset["train"].map(prepare_dataset, num_proc=2)
raw_dataset["val"] = raw_dataset["val"].cast_column("audio", Audio(sampling_rate=16000))
raw_dataset["val"] = raw_dataset["val"].map(prepare_dataset, num_proc=2)
raw_dataset["test"] = raw_dataset["test"].cast_column("audio", Audio(sampling_rate=16000))
raw_dataset["test"] = raw_dataset["test"].map(prepare_dataset, num_proc=2)

# Concatenate cv_swahili_train and cv_swahili_validate
combined_train_validate = concatenate_datasets([raw_dataset["train"], raw_dataset["val"]])

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



model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-Base-960h",
    mask_time_prob=0.75,
    mask_time_length=10,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_encoder()

training_args = TrainingArguments(
  output_dir='/home2/meenakshi.sirigiraju/data/hugg_models',
  group_by_length=True,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=400,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=100,
  logging_steps=5,
  #optim="adamw_bnb_8bit",
  gradient_accumulation_steps=2,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=500,
  save_total_limit=2,
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["val"],
    tokenizer=processor.feature_extractor,
)

#optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
#total_steps = 1000
#scheduler = CosineAnnealingLR(
#    optimizer,
#    T_max=total_steps,
#    eta_min=0,  # Minimum learning rate
#)

#def update_lr():
#  scheduler.step()

print("Model training started !!!!")
#for epoch in range(training_args.num_train_epochs):
 #   trainer.train()
  #  update_lr()
    #trainer.evaluate()

#print("Model training started !!!!")
trainer.train()

print("Completted !!!!")

print("Evaluate on test  ")

model.save_pretrained('/home2/meenakshi.sirigiraju/data/hugg_models/save_model')
tokenizer.save_pretrained('/home2/meenakshi.sirigiraju/data/hugg_models/save_model')
trainer.save_model('/home2/meenakshi.sirigiraju/data/hugg_models/save_model')

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
print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
