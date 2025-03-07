import wikipedia
import re
from collections import Counter
import math

def tokenize(text):
    text = text.lower()  
    text = re.sub(r'[^a-z\s\1-9]', '', text)  
    tokens = text.split() 
    return tokens

def create_bg(tokens):
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

def count_bg(bgs):
    return Counter(bgs)

def count_tokens(tokens):
    return Counter(tokens)

def calc_bg(bg_counts, ug_counts, vocab_size):
    bg_probabilities = {}
    for bg, count in bg_counts.items():
        bg_probabilities[bg] = (count + 1) / (ug_counts[bg[0]] + vocab_size)  # Laplace Smoothing
    return bg_probabilities

def calculate_perplexity(sentence, bigram_counts, unigram_counts, vocab_size):
    tokens = tokenize(sentence)
    bigrams = create_bg(tokens)

    log_prob_sum = 0
    N = len(tokens)

    for bigram in bigrams:
        bigram_count = bigram_counts.get(bigram, 0)
        unigram_count = unigram_counts.get(bigram[0], 0)
        
        # Apply Laplace smoothing
        prob = (bigram_count + 1) / (unigram_count + vocab_size)
        
        log_prob_sum += math.log(prob)

    perplexity = math.exp(-log_prob_sum / N)
    return perplexity

# Fetch Wikipedia text
topic = "Penguins of Madagascar" 
text = wikipedia.page(topic).content  # Get full Wikipedia article

words = text.split()[:1000]
training_text = " ".join(words)  

print("Penguins of Madagascar")
print(training_text)
print("\n" + "="*50 + "\n")

tokens = tokenize(training_text)
print("Tokens:", tokens[:10], ("..."))  

ug_counts = count_tokens(tokens)
print("Token Counts:", list(ug_counts.items())[:10], ("..."))  

bgs_list = create_bg(tokens)
print("Bi-grams:", bgs_list[:10], ("...")) 

bg_counts = count_bg(bgs_list)
print("Bi-gram Counts:", list(bg_counts.items())[:10], "...")  

# Use Laplace Smoothing
bg_probabilities = calc_bg(bg_counts, ug_counts, vocab_size=len(set(tokens)))

print("\nBi-gram Probabilities:")
for bg, prob in list(bg_probabilities.items())[:10]: 
    count_bg = bg_counts[bg]
    count_ug = ug_counts[bg[0]]  
    fraction = f"{count_bg}/{count_ug}" if count_ug > 0 else f"{count_bg}/0"
    print(f"P({bg[1]} | {bg[0]}) = {fraction} = {prob:.4f}")

print("...")

# Test sentence
test_sentence = "The quick brown fox jumps over the lazy dog near the bank of the river."

# Calculate
perplexity_score = calculate_perplexity(test_sentence, bg_counts, ug_counts, vocab_size=len(set(tokens)))

# Display test sentence and score
print(f"\nTest Sentence: \"{test_sentence}\"")
print(f"Perplexity of the test sentence: {perplexity_score}")
