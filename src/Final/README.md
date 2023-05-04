¬¬	This final project incorporates several of the AI techniques learned throughout this course. This project is an abstraction of these techniques as an elf travels the map of his fantasy world. As the player travels, they lose money from their budget. They may also encounter enemies. If ever their budget or health hits zero, the player loses. The player wins by reaching the destination city. The map for these cities is randomly generated with each execution of the program. This includes the placement of the cities, the ten routes connecting cities, and the landscape overall. Finally, this game is also able to be played by a non-human (AI) player. Furthermore, this project demonstrates an understanding of the concepts of artificial intelligence learned in class as well as various libraries and packages that are available in Python for AI. It also demonstrates learning of how to build out a system with modularity and integration of these modules into a functioning system. Discussed in this report are also the conflicts I ran into accompanied by how I went about solving these issues. While having a functioning project is enjoyable, the most beneficial part of any project is the knowledge gained from the process. An additional section I am including in this report pertains to retrospection. I find it valuable to look back at both a product and its process and analyze to learn. As a result, I am incorporating my thoughts as to what I would have done differently and how my initial expectations align with my project.
¬¬	This project consists of three primary AI components. While other components may be present within the project, they are not as central to the project. The key components include AI intelligence agents, a genetic algorithm, an AI player, and an AI text generator; all of these components were used to solve problems. The intelligence agents are represented through the game’s combat events. The user could be facing any one of the three agents programmed. These agents are referred to as the Mimic agent, Switch agent, and Single agent. They each employ their own unique strategy in battle. This allowed the game to have unpredictable turn-based combat. Without these agents, combat would have no strategy. It would be either truly random or predetermined. The genetic algorithm was used to optimize the generation of cities on the landscape. This allowed me to take a generated layout of cities, apply it into another generated layout, and analyze the result. Repeating this process leads to an optimized layout. The landscape was formed through the use of Perlin noise; this is one of many utilized Python packages. The AI player is capable of performing any action a human player can perform. The AI player is trained and takes appropriate action when faced with combat. The player is not trained on the best way to move on the map or optimize its budget. Having an AI player solves the issue of human biases. An AI controlled player must be trained on nothing but the game itself and utilizes its knowledge to play the game to the best of its understanding. This also allowed for testing of the project that did not demand input that relied on human decision. Last is the AI text generation. This generation is through GPT-2. I implemented this to create more of a personal story with the user as they play the game. The text generator documents the character’s journey from city to city and lets us hear the character’s inner thoughts as they reflect. With AI text generation, each story is new, unique, and fresh to the user. No story will be repeated. No story is predetermined or hard-coded.
¬¬	Other problems I ran into were solved through internet searches and conversations with chat-GPT. Consulting an AI resource about AI helped me understand both the issue at hand and AI itself. In respect to the contents of the project. Chat-GPT assisted me in writing and debugging code as well as determining ideas. It exposed me to libraries I found useful. But, when it strayed too far from what I needed, it taught me to remember to consult myself first. Sometimes, an issue was already solved by code I had previously written. 
This leads me into my reflection section. Looking back at this project, I wish I had a better understanding of how all of my modules worked. Without fully knowing each part of each module I worked with, it became difficult to know what could and could not be merged. I found myself scrolling through all of my labs and ended up only using a few of my previous labs. The solution I found may have worked best for myself, but I know there could be plenty of other solutions that use parts of my previous modules I did not touch. I would love to see how I would revisit this task now that I have completed it. Another lesson learned is how there are multiple ways to accomplish the same task. While I found myself trying to force one solution, I often had to start over as it became overly complicated. When restarting with a new perspective, a new foundation. I found the original solution only required a few lines of code. When analyzing my first impressions, I found I thought each and every lab would serve its purpose. Looking at what I have created, I did not need all of these modules. I can say that there were labs I did not need for this project, but I can only wonder how much better my project could be if I found a way to implement them all. 

    
Appendix
https://huggingface.co/nvidia/GPT-2B-001

https://chat.openai.com/?model=text-davinci-002-render-sha
Help me write a python function to see if a route is valid, routes is set up like this [[[x1,y1],[x2,y2]]] it is a numpy array
ChatGPT
Sure, here's an example Python function that checks if a route is valid based on the format you provided:
python
import numpy as np

def is_valid_route(route):
    """
    Determines if a route is valid based on the format [[[x1,y1],[x2,y2]]].
    
    Args:
    route (numpy array): The route to check.
    
    Returns:
    bool: True if the route is valid, False otherwise.
    """
    # Check if the route is a numpy array
    if not isinstance(route, np.ndarray):
        return False
    
    # Check if the route has the correct shape
    if route.shape != (1, 2, 2):
        return False
    
    # Check if each point in the route is a valid coordinate
    for i in range(2):
        for j in range(2):
            if not isinstance(route[0][i][j], (int, float)):
                return False
    
    # All checks passed, so the route is valid
    return True
You can use this function to check if a given route is valid by passing the route as a numpy array to the is_valid_route() function. The function returns True if the route is valid, and False otherwise.
  
Check the list of routes to see if it contains a start and end that corresponds to two cities
ChatGPT
Sure! Here's an example Python function that checks a list of routes to see if it contains a start and end that correspond to two cities:
python
def has_start_and_end(routes, start_city, end_city):
    """
    Checks if a list of routes contains a start and end that correspond to two cities.
    
    Args:
    routes (list): The list of routes to check.
    start_city (str): The name of the starting city.
    end_city (str): The name of the ending city.
    
    Returns:
    bool: True if the list of routes contains a start and end that correspond to the two cities, False otherwise.
    """
    for route in routes:
        # Check if the route has the correct format
        if is_valid_route(route):
            # Check if the starting and ending points of the route correspond to the two cities
            if get_city_name(route[0][0]) == start_city and get_city_name(route[0][1]) == end_city:
                return True
    # No valid route was found that corresponds to the two cities
    return False

def is_valid_route(route):
    """
    Determines if a route is valid based on the format [[[x1,y1],[x2,y2]]].
    
    Args:
    route (numpy array): The route to check.
    
    Returns:
    bool: True if the route is valid, False otherwise.
    """
    # Check if the route is a numpy array
    if not isinstance(route, np.ndarray):
        return False
    
    # Check if the route has the correct shape
    if route.shape != (1, 2, 2):
        return False
    
    # Check if each point in the route is a valid coordinate
    for i in range(2):
        for j in range(2):
            if not isinstance(route[0][i][j], (int, float)):
                return False
    
    # All checks passed, so the route is valid
    return True

def get_city_name(coordinates):
    """
    Gets the name of the city that corresponds to the given coordinates.
    
    Args:
    coordinates (list): The coordinates of the city.
    
    Returns:
    str: The name of the city.
    """
    # Replace this with your own code to map coordinates to city names
    # For example, you could use a dictionary where the keys are the coordinates and the values are the city names
    return "City " + str(coordinates)

You can use the has_start_and_end() function by passing it the list of routes, the name of the starting city, and the name of the ending city. The function will return True if the list of routes contains a valid route that corresponds to the two cities, and False otherwise.
Note that the get_city_name() function in this example is a placeholder function that you would need to replace with your own code to map coordinates to city names. You could use a dictionary where the keys are the coordinates and the values are the city names, or you could use a reverse geocoding API to lookup the city name based on the coordinates.
  
when you create this function, the parameters must be a numpy array (of city IDs), the current city you are at, and the target city. return whether there is a route between the current city and the target city
ChatGPT
Sure, here's an example Python function that takes in a numpy array of city IDs, the current city, and the target city, and returns whether there is a route between the current city and the target city:
python
import numpy as np

def has_route(city_ids, current_city, target_city):
    """
    Checks if there is a route between the current city and the target city.
    
    Args:
    city_ids (numpy array): The numpy array of city IDs.
    current_city (int): The ID of the current city.
    target_city (int): The ID of the target city.
    
    Returns:
    bool: True if there is a route between the current city and the target city, False otherwise.
    """
    # Use breadth-first search to find a route between the current city and the target city
    queue = [current_city]
    visited = set()
    
    while queue:
        # Get the next city in the queue
        current = queue.pop(0)
        
        # Check if we've already visited this city
        if current in visited:
            continue
        
        # Check if this is the target city
        if current == target_city:
            return True
        
        # Add all neighboring cities to the queue
        neighbors = get_neighbors(city_ids, current)
        queue.extend(neighbors)
        
        # Mark this city as visited
        visited.add(current)
    
    # No route was found between the current city and the target city
    return False

def get_neighbors(city_ids, current_city):
    """
    Gets the neighboring cities of the current city in the numpy array of city IDs.
    
    Args:
    city_ids (numpy array): The numpy array of city IDs.
    current_city (int): The ID of the current city.
    
    Returns:
    list: The IDs of the neighboring cities of the current city.
    """
    # Find all the routes that start from the current city
    routes = np.where(city_ids[:, 0] == current_city)[0]
    
    # Extract the IDs of the neighboring cities
    neighbors = city_ids[routes, 1].tolist()
    
    return neighbors
You can use this function by passing it the numpy array of city IDs, the ID of the current city, and the ID of the target city. The function will return True if there is a route between the current city and the target city, and False otherwise.
Note that the get_neighbors() function in this example extracts the IDs of the neighboring cities by searching for all the routes that start from the current city in the numpy array of city IDs. You would need to replace this function with your own code that extracts the neighboring cities based on your specific data structure.
Write me a function to get a specific route based on the current list of routes, the current city, and the desired city
ChatGPT
Sure! Here's an example function that finds a specific route based on the current list of routes, the current city, and the desired city:
python
import numpy as np

def get_route(routes, current_city, desired_city, cities):
    """
    Returns the route from the current city to the desired city in the given routes.
    
    Args:
    routes (numpy array): The numpy array of routes.
    current_city (int): The ID of the current city.
    desired_city (int): The ID of the desired city.
    cities (numpy array): The numpy array that maps city IDs to their coordinates.
    
    Returns:
    numpy array or None: The route from the current city to the desired city, or None if no such route is found.
    """
    current_coords = cities[current_city]
    desired_coords = cities[desired_city]
    for route in routes:
        if np.array_equal(route[:2], [current_coords, desired_coords]) or \
           np.array_equal(route[:2], [desired_coords, current_coords]):
            return route
    return None
This function takes in the routes numpy array, the ID of the current_city, the ID of the desired_city, and the cities numpy array that maps city IDs to their coordinates. It first gets the coordinates of the current_city and desired_city from the cities array. Then, it loops over each route in the routes array, and checks if the first two elements of the route are equal to the numpy arrays [current_coords, desired_coords] or [desired_coords, current_coords]. If such a route is found, the function returns that route. If no such route is found, the function returns None.
  
2 / 2
refer back to the valid route program. in your function, know all routes are bidirectional
ChatGPT
If all routes are bidirectional, then you can simplify the ValidRoute function by removing the second condition in the if statement:
python
import numpy as np

def is_valid_route(routes, current, target):
    """
    Checks if there is a route between the current city and the target city in the given routes.
    
    Args:
    routes (numpy array): The numpy array of routes.
    current (int): The ID of the current city.
    target (int): The ID of the target city.
    
    Returns:
    bool: True if there is a route between the current city and the target city in the given routes, False otherwise.
    """
    for route in routes:
        if np.array_equal(route[:2], [current, target]) or \
           np.array_equal(route[:2], [target, current]):
            return True
    return False
This version of the function takes in the routes numpy array, the ID of the current city, and the ID of the target city. It loops over each route in the routes array, and checks if the first two elements of the route are equal to the numpy arrays [current, target] or [target, current]. If such a route is found, the function returns True. If no such route is found, the function returns False.
  
numpy absolute value
ChatGPT
In numpy, you can take the absolute value of an array or a specific element in an array using the numpy.absolute() function. This function returns the absolute value of each element in the input array.
Here's an example usage:
python
import numpy as np

arr = np.array([-2, -1, 0, 1, 2])
abs_arr = np.absolute(arr)
print(abs_arr)
Output:
csharp
[2 1 0 1 2]
In this example, we create a numpy array arr with a mix of negative and positive values. We then apply the numpy.absolute() function to arr and store the result in abs_arr. The function returns a new numpy array where each element is the absolute value of the corresponding element in arr.
Note that the numpy.absolute() function also has an alias called numpy.abs(), which can be used interchangeably.
  
How do I use huggingface to incorporate ai text generation into my python project
ChatGPT
To use Hugging Face to incorporate AI text generation into your Python project, you can use the transformers library which provides pre-trained models for various natural language processing (NLP) tasks, including text generation.
Here are the basic steps to get started:
1.	Install the transformers library:
1.	pip install transformers
2.	
3.	Choose a pre-trained language model to use for text generation. Hugging Face provides a number of pre-trained models which can be accessed through their model hub.
4.	Load the language model using the AutoModelForCausalLM class from the transformers library. This class loads a pre-trained language model that can generate text one token at a time in a causal way.
5.	Tokenize the input text using the tokenizer of the pre-trained language model. The tokenizer converts the input text into a sequence of tokens that can be processed by the language model.
6.	Use the language model to generate new text based on the input sequence of tokens. The generate() method of the loaded language model generates new text based on the input sequence of tokens.
Here's an example code snippet that demonstrates how to use the transformers library to generate text using a pre-trained language model:
python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained language model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate new text
prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
In this example, we first load the pre-trained GPT-2 language model using its name (gpt2). We then provide a prompt text The quick brown fox and convert it into a sequence of input tokens using the tokenizer.encode() method. Finally, we use the model.generate() method to generate new text based on the input sequence of tokens. The generated text is decoded using the tokenizer and printed to the console.
Note that the generate() method takes several arguments such as max_length and do_sample that control the behavior of the text generation process. You can refer to the Hugging Face documentation for more information on these arguments and their values.
  
what would be an appropriate prompt for arriving to a new city
ChatGPT
Here are some possible prompts that you could use for arriving to a new city:
1.	"I just arrived in [city name], what should I check out first?"
2.	"What are some must-see sights or experiences in [city name]?"
3.	"What are some of the best local food or drink recommendations in [city name]?"
4.	"What are some hidden gems or off-the-beaten-path places to explore in [city name]?"
5.	"What are some fun or unique activities to do in [city name] that locals love?"
These prompts can be tailored to your specific interests or goals for your visit. For example, if you're a foodie, you might focus on asking for recommendations on local cuisine. If you're interested in history, you might ask for information about historic sites or museums. The important thing is to be specific about what you're interested in and ask for recommendations or advice from locals or people familiar with the city.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results. Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
ChatGPT
This message is a warning from the Hugging Face Transformers library, indicating that you have not explicitly set the attention_mask or pad_token_id when calling a text generation method.
The attention_mask is an optional parameter that specifies which tokens in the input sequence should be attended to by the model, and which should be ignored. By default, the attention_mask is set to None, which means that all tokens are attended to. However, in some cases, you may want to specify a custom attention mask to exclude certain tokens, such as padding tokens or special tokens like [SEP] or [CLS].
The pad_token_id is a special token ID that is used to pad sequences to a fixed length during training or inference. By default, the pad_token_id is set to 0, which corresponds to the [PAD] token in most pre-trained language models. However, you may want to set a different pad token ID if your input sequences use a different padding token.
To address the warning, you can pass in an appropriate attention_mask and pad_token_id parameter when calling the text generation method. For example:
makefile
input_ids = tokenizer.encode(prompt, return_tensors="pt")
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
generated_text = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
In this example, we create an attention mask tensor with the same shape as the input tensor (which has shape [batch_size, sequence_length]), and set all values to 1 to indicate that all tokens should be attended to. We also set the pad_token_id to the end-of-sequence (EOS) token ID, which is obtained from the tokenizer.
  
i want to print the transformation of the generated text
ChatGPT
If you want to print the transformation of the generated text, you can use the tokenizer.decode() method to convert the generated token IDs back into text.
For example:
scss
generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated text:", generated_text)
In this example, we first generate the token IDs for the generated text using the model.generate() method, with the same input_ids, attention_mask, and pad_token_id parameters as before. We then use the tokenizer.decode() method to convert the generated token IDs back into text, with the skip_special_tokens=True argument to remove any special tokens (like [CLS] or [SEP]) that may have been added by the model.
Finally, we print out the generated text using the print() function. This should show you the transformed version of the generated text.
  
so how do i implement all of this?
ChatGPT
Here's an example of how you can implement the code I provided earlier to generate text using Hugging Face:
python
from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead

# load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

# set the input text and generate new text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
generated_ids = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=1)

# decode the generated text and print it
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated text:", generated_text)
In this example, we first load the tokenizer and model using Hugging Face's AutoTokenizer and AutoModelWithLMHead classes. We then set the input text to "Once upon a time" and encode it using the tokenizer.encode() method.
We then generate new text using the model.generate() method, with the input_ids parameter set to the encoded input text, and the max_length and num_return_sequences parameters set to control the length and number of generated texts.
Finally, we decode the generated text using the tokenizer.decode() method, with the skip_special_tokens=True argument to remove any special tokens, and print the resulting text using the print() function.
  
ValueError: The following `model_kwargs` are not used by the model: ['pad_tokenid'] (note: typos in the generate arguments will also show up in this list)
ChatGPT
This error message suggests that the model_kwargs argument passed to the generate() method of the Hugging Face model is not recognized by the model. In particular, it indicates that the pad_tokenid argument is not being used by the model.
One potential reason for this error is a typo in the argument name. Make sure that you are using the correct argument name (i.e., pad_token_id instead of pad_tokenid).
If you are still encountering this error, you can try removing the model_kwargs argument altogether, as it is not strictly necessary for generating text with Hugging Face models.

