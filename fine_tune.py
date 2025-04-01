import os
import fitz  # PyMuPDF for PDF extraction
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

# ✅ Force CPU execution
os.environ["ACCELERATE_NOT_FOUND"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Prevents GPU usage
device = torch.device("cpu")

# ✅ Load GPT-Neo-1.3B model
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(device)  # ✅ Ensure CPU execution

tokenizer.pad_token = tokenizer.eos_token  # ✅ Set padding token

# ✅ Extract text from PDFs in 'data/' folder
def extract_text_from_pdfs(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with fitz.open(pdf_path) as doc:
                text = "\n".join([page.get_text("text") for page in doc])
                texts.append(text)
    return texts

pdf_texts = extract_text_from_pdfs("data/")

# ✅ Convert extracted text into a dataset
dataset = Dataset.from_dict({"text": pdf_texts})

# ✅ Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ✅ Training arguments (CPU Optimized)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # ✅ Small batch size for CPU
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    evaluation_strategy="no",
    report_to="none",
    no_cuda=True,  # ✅ Ensures CPU-only execution
    fp16=False,    # ✅ Disable mixed precision (only needed for GPUs)
)

# ✅ Trainer setup
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ✅ Train model on extracted text
trainer.train()
print("Fine-tuning complete!")

# ✅ Save the fine-tuned model
model.save_pretrained("./fine_tuned_EleutherAI")
tokenizer.save_pretrained("./fine_tuned_EleutherAI")

print("🎯 Fine-tuning complete! Model saved to './fine_tuned_EleutherAI'")