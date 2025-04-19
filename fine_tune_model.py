# fine_tune_model_4k.py
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import transformers

transformers.logging.set_verbosity_error()

# Load dataset
dataset_path = "C:/Users/HP/Desktop/princebot/princebot/model/subset_dataset.json"
dataset = load_dataset('json', data_files=dataset_path)['train']
print(f"Dataset loaded with {len(dataset)} entries")

# Split dataset (80/20 split)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Load tokenizer and model
model_name = "google/flan-t5-base"
cache_dir = "C:/Users/HP/Desktop/hf_cache"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True, cache_dir=cache_dir)
model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)

# Tokenize dataset
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    labels = tokenizer(targets, max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

# Training arguments
training_args = TrainingArguments(
    output_dir="C:/Users/HP/Desktop/princebot/princebot/model/t5_finetuned_4k",
    num_train_epochs=5,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='C:/Users/HP/Desktop/princebot/princebot/model/logs_4k',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save
model.save_pretrained("C:/Users/HP/Desktop/princebot/princebot/model/t5_finetuned_4k")
tokenizer.save_pretrained("C:/Users/HP/Desktop/princebot/princebot/model/t5_finetuned_4k")
print("✅ Fine-tuning completed")