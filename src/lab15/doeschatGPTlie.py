# The meaning of life cannot be summed up in one sentence, as it is a complex and multifaceted question that has no universally accepted answer.

# regen: The meaning of life cannot be encapsulated in a single sentence as it is a complex and deeply personal question that varies from person to person.

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer, util 

def translate_response():
   
   # util.pytorch_cos_sim(embedding_1, embedding_2)  

    # Define the sentence transformer model
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Define the list of sentences
    sentences = ["The meaning of life cannot be summed up in one sentence, as it is a complex and multifaceted question that has no universally accepted answer.", 
            "The meaning of life cannot be encapsulated in a single sentence as it is a complex and deeply personal question that varies from person to person."]

# Split the sentences into individual sentences
    sentences = [sentence.split('. ') for sentence in sentences]

# Set the similarity threshold
    threshold = 0.5

# Calculate the similarity score for each pair of sentences and store in a list
    similarity_scores = []
    for i in range(len(sentences[0])):
        for j in range(len(sentences[1])):
        # Encode the two sentences using the model
            sentence_embeddings = model.encode([sentences[0][i], sentences[1][j]])

        # Calculate the cosine similarity between the sentence embeddings
            similarity_score = util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])

        # Add the similarity score to the list
            similarity_scores.append(similarity_score)

# Check if any of the similarity scores exceed the threshold
    if any(similarity_score > threshold for similarity_score in similarity_scores):
        print("There are pairs of sentences with similarity scores above the threshold.")
    else:
        print("No pairs of sentences have similarity scores above the threshold.")

 