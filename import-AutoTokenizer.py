from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
model = AutoModelForSeq2SeqLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")


tokenizer.save_pretrained("./models/stabilityai/stablelm-tuned-alpha-7b")
model.save_pretrained("./models/stabilityai/stablelm-tuned-alpha-7b")


### Now when you’re offline, reload your files with PreTrainedModel.from_pretrained() from the specified directory:

# Copied
# tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
# model = AutoModel.from_pretrained("./your/path/bigscience_t0")