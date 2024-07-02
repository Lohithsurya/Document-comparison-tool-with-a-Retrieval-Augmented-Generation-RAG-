from transformers import AutoModel, AutoTokenizer

class HuggingFaceEmbeddings:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L12-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.batch_size = 32  # Maximum batch size allowed by Chroma

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            encoded_input = self.tokenizer(batch, return_tensors="pt", padding=True)
            output = self.model(**encoded_input)[0][:, 0, :]
            embeddings.extend(output.detach().numpy().tolist())
        return embeddings

    def embed_query(self, query):
        encoded_input = self.tokenizer(query, return_tensors="pt", padding=True)
        output = self.model(**encoded_input)[0][:, 0, :]
        return output.detach().numpy().tolist()[0]

def get_embedding_function():
    return HuggingFaceEmbeddings()
