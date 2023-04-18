# The meaning of life cannot be summed up in one sentence, as it is a complex and multifaceted question that has no universally accepted answer.

# regen: The meaning of life cannot be encapsulated in a single sentence as it is a complex and deeply personal question that varies from person to person.

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

def translate_response():
    sentences = ["The meaning of life cannot be summed up in one sentence, as it is a complex and multifaceted question that has no universally accepted answer.", 
                "The meaning of life cannot be encapsulated in a single sentence as it is a complex and deeply personal question that varies from person to person."]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L7-v2')

    embedding_1 = model.encode(sentences[0], convert_to_tensor = True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor = True)

    print(embedding_1)
    print(embedding_2)
   # util.pytorch_cos_sim(embedding_1, embedding_2)  