import wikipedia
import re
from collections import Counter

def tokenize(text):
    text = text.lower()  
    text = re.sub(r'[^a-z\s]', '', text)  # Remove anything that isn't a-z or space
    tokens = text.split() 
    unique_tokens = list(set(tokens))
    return tokens, unique_tokens

def create_bg(tokens):
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

def count_bg(bgs):
    return Counter(bgs)

def count_tokens(tokens):
    return Counter(tokens)

def calc_bg(bg_counts, ug_counts):
    bg_probabilities = {}
    for bg, count in bg_counts.items():
        bg_probabilities[bg] = count / ug_counts[bg[0]] 
    return bg_probabilities

# Fetch Wikipedia text
topic = "Penguins of Madagascar" 
text = wikipedia.page(topic).content  # Get full Wikipedia article

# Extract first 1000 words
words = text.split()[:1000]
training_text = " ".join(words) 

print("Penguins of Madagascar")
print(training_text)
print("\n" + "="*50 + "\n")

tokens, unique_tokens = tokenize(training_text)
print("Unique Tokens:", unique_tokens)


ug_counts = count_tokens(tokens)

bgs_list = create_bg(tokens)
bg_counts = count_bg(bgs_list)
print("\nBi-gram Counts:")
for bg, count in list(bg_counts.items())[:10]: 
    print(f"{bg}: {count}")
print("...")

bg_probabilities = calc_bg(bg_counts, ug_counts)
print("\nBi-gram Probabilities:")
for bg, prob in list(bg_probabilities.items())[:10]: 
    count_bg = bg_counts[bg]
    count_ug = ug_counts[bg[0]]  
    fraction = f"{count_bg}/{count_ug}" 
    print(f"P({bg[1]} | {bg[0]}) = {fraction} = {prob:.4f}")
print("...")

# Text Prediction
def predict_next_word(bigram_probs, current_word):
    candidates = {k[1]: v for k, v in bigram_probs.items() if k[0] == current_word}
    if not candidates:
        return None  
    return max(candidates, key=candidates.get)

# Text Generation
def generate_bigram_text(bigram_model, start_word, length=10):
    current_word = start_word.lower()
    generated_text = [current_word]
    for _ in range(length):
        candidates = {k[1]: v for k, v in bigram_model.items() if k[0] == current_word}
        if candidates:
            current_word = max(candidates, key=candidates.get)
        else:
            break 
        generated_text.append(current_word)
    return " ".join(generated_text)

task = input("Would you like to (1) Predict the next word or (2) Generate text? Enter 1 or 2: ")

if task == "1":
    # Text Prediction
    current_word = input("Enter the word you want to predict the next word for: ")
    predicted_word = predict_next_word(bg_probabilities, current_word)
    if predicted_word:
        print(f"Predicted next word after '{current_word}': {predicted_word}")
    else:
        print(f"No prediction available for '{current_word}'.")

elif task == "2":
    # Text Generation
    start_word = input("Enter the starting word for text generation: ")
    length = int(input("Enter the length of the generated text: "))
    generated_text = generate_bigram_text(bg_probabilities, start_word, length)
    print(f"Generated Text with the length of {length} is: ", generated_text)

else:
    print("Invalid choice! Please enter 1 or 2.")
