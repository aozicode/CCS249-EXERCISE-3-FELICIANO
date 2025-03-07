import wikipedia
import re
from collections import Counter

def tokenize(text):
    text = text.lower()  
    text = re.sub(r'[^a-z\s0-9]', '', text) 
    tokens = text.split() 
    return tokens

def create_grams(tokens, n=3):
    """Create n-grams (trigrams here)"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def count_grams(grams):
    return Counter(grams)

def count_tokens(tokens):
    return Counter(tokens)

def calc_grams(gram_counts, bg_counts):
    gram_probabilities = {}
    for gram, count in gram_counts.items():
        bigram = gram[:2]  
        if bg_counts[bigram] > 0: 
            gram_probabilities[gram] = count / bg_counts[bigram]
        else:
            gram_probabilities[gram] = 0  
    return gram_probabilities

def calculate_perplexity(sentence, trigram_counts, bigram_counts, vocab_size):
    tokens = tokenize(sentence)
    trigrams = create_grams(tokens)

    perplexity = 1
    N = len(tokens)
    
    for trigram in trigrams:
        bigram = trigram[:-1]
        
        trigram_count = trigram_counts[trigram] if trigram in trigram_counts else 0
        bigram_count =bigram_counts[bigram] if bigram in bigram_counts else 0
        prob = (trigram_count + 1) / (bigram_count + vocab_size)
        perplexity *= prob

    perplexity = perplexity ** (-1/N)  
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

ug_counts = count_tokens(tokens)

tgs_list = create_grams(tokens, n=3)

tg_counts = count_grams(tgs_list)

bgs_list = create_grams(tokens, n=2)
bg_counts = count_grams(bgs_list)

tg_probabilities = calc_grams(tg_counts, bg_counts)

print("\nTokens:", tokens[:5], "...")
print("\nToken Counts:")
for token, count in list(ug_counts.items())[:5]:
    print(f"{token}: {count}")
print ("...")

print("\nTri-gram Counts:")
for tg, count in list(tg_counts.items())[:5]:
    print(f"{tg}: {count}")
print ("...")

print("\nTri-gram Probabilities:")
for tg, prob in list(tg_probabilities.items())[:5]:
    count_tg = tg_counts[tg]
    count_bg = bg_counts[tg[:2]]  
    fraction = f"{count_tg}/{count_bg}" if count_bg > 0 else f"{count_tg}/0"
    print(f"P({tg[2]} | {tg[0]}, {tg[1]}) = {fraction} = {prob:.4f}")
print("...")

# Test sentence
test_sentence = "The quick brown fox jumps over the lazy dog near the bank of the river."

perplexity_score = calculate_perplexity(test_sentence, tg_probabilities, bg_counts, vocab_size=len(set(words)))

print(f"\nTest Sentence: \"{test_sentence}\"")
print(f"Perplexity of the test sentence: {perplexity_score}")
