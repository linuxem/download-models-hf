from transformers import AutoModel, AutoTokenizer

# Do this on a machine with internet access
model = AutoModel.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")

_ = model.save_pretrained("/home/eli/StableLM/modules")
_ = tokenizer.save_pretrained("/home/eli/StableLM/modules")


#model = AutoModel.from_pretrained("/home/eli/StableLM/models")
#tokenizer = AutoTokenizer.from_pretrained("/home/eli/StableLM/models")
