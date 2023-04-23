from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

#question1: "what question can I ask you and receive two different responses"
#question2: "What is the capital of France?"
#question3: "What is a word similar to devolve that begins with b?"




# Input paragraphs
# 
# paragraph1 is the first response
# paragraph2 is the regenerated response
#
# 
#paragraph1 = "As an AI language model, I strive to provide consistent and accurate responses to your questions. However, there are some questions that could potentially generate different responses based on the context or the way the question is phrased. For example: What is the meaning of life? Depending on your personal beliefs, values, and experiences, you may receive different responses to this question. Is it ethical to use animals in scientific research? This is a controversial topic, and different people may have different opinions on the ethics of using animals in scientific research. What is the best programming language for beginners? There are many programming languages, and different people may have different opinions on which one is the best for beginners based on their personal experience and expertise. In these cases, it's important to remember that different perspectives and opinions can exist and that it's okay to have differing views."
#paragraph2 = "Here's an example of a question you can ask me and receive two different responses: Question: \"What is the capital of France?\" Response 1: \"The capital of France is Paris.\" Response 2: \"The capital of France is Marseille.\" In this example, the two different responses are conflicting because Paris is the correct capital of France, but Marseille is not. Depending on the specific context or knowledge cutoff date of the language model, it may generate different responses, which could include incorrect or outdated information. It's always a good practice to verify information from reliable sources to ensure accuracy."
#
#
#paragraph1 = "The capital of France is Paris."
#paragraph2 = "Response 1: \"The capital of France is Paris.\" Response 2: \"The capital of France is Marseille.\""
#
#
paragraph1 = "One word that is similar to \"devolve\" and begins with \"b\" is \"deteriorate.\" It means to become worse or to decline in quality or value over time."
paragraph2 = "A word similar to \"devolve\" that begins with \"b\" is \"bifurcate.\""

# Split paragraphs into sentences
sentences1 = paragraph1.split(". ")
sentences2 = paragraph2.split(". ")

# Compute sentence embeddings for each sentence
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine similarity between sentences
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

# Compute L2 distance between sentences, this is the metric of choice
l2_distances = np.sqrt(np.sum((embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0))**2, axis=-1))

# Normalize L2 distances to be between 0 and 1
l2_distances = l2_distances / np.sqrt(embeddings1.shape[-1])

# Print the similarity score for each sentence pair
# Cosine is used to maintain a number between 0 and 1.
# Cosine is best as the cosine of zero degrees is 1. Sine would require manipulation.
# typically, an L2 distance score between 0.5 and 1 would be deemed similar, but the user can always define their own threshold. 
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print("Sentence 1:", sentences1[i])
        print("Sentence 2:", sentences2[j])
        print("Similarity score:", cosine_scores[i][j].item())
        print("L2 distance:", l2_distances[i][j].item())
