'''

The model (`sentence-transformers/all-MiniLM-L12-v2`) is trained to create sentence embeddings by combining information from all the words in a sentence using a method called mean pooling — meaning the sentence embedding should be the average of all token embeddings.

The older version of the code instead took only the first token (CLS token) and treated it as the whole sentence representation using this line

    `output = self.model(**encoded_input)[0][:, 0, :]`

This is not how this model is meant to be used, so the embeddings may not capture the full meaning of the text. There was also zero normalization being done. Adding normalization generates more accurate similarity scores.

The older version of the code also manually handled batching, tokenization and converting outputs into usable formats. The newer approach using LangChain’s `HuggingFaceEmbeddings` is better because it handles all of these details internally. 

'''

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},  # or "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings
