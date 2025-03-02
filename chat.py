import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import wikipediaapi
import wolframalpha
from transformers import pipeline
import torch
import requests

# 🔹 Load chatbot model
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# 🔹 Wolfram Alpha API Setup (for math & science)
WOLFRAM_ALPHA_APP_ID = "9QKVTT-55HYYQ8E9K"
wolfram_client = wolframalpha.Client(WOLFRAM_ALPHA_APP_ID)

# 🔹 Wikipedia API Setup (for general knowledge)
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="MyChatbot/1.0 (https://yasuo72.github.io/; kingrohi37@gmail.com)"
)

# 🔹 AI Model for General Chat
generator = pipeline(
    'text-generation',
    model='microsoft/DialoGPT-medium',
    device=0 if torch.cuda.is_available() else -1,
    max_length=80,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.5
)


# 🔹 Enhanced NLP Processing
def enhanced_clean_text(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    keep_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN',
                 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
    filtered = [word for word, tag in tagged if tag in keep_tags]
    return [lemmatizer.lemmatize(word.lower()) for word in filtered]


# 🔹 Bag of Words Representation
def context_aware_bow(sentence):
    sentence_words = enhanced_clean_text(sentence)
    bag = [0] * len(words)
    for i, w in enumerate(words):
        bag[i] = 1 if w in sentence_words else 0
    return np.array(bag)


# 🔹 Predict Intent
def predict_class(sentence, threshold=0.25):
    p = context_aware_bow(sentence)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# 🔹 Wolfram Alpha Query (Math & Science)
def ask_wolfram(question):
    try:
        res = wolfram_client.query(question)
        answer = next(res.results).text
        return answer
    except Exception:
        return None


# 🔹 Wikipedia Search (Definitions & Facts)
def search_wikipedia(query):
    try:
        page = wiki_wiki.page(query)
        if page.exists():
            return page.summary[:300] + "..."
        return None
    except Exception:
        return None


# 🔹 DuckDuckGo API for Real-Time Search
def search_duckduckgo(query):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url)
        data = response.json()
        if 'AbstractText' in data and data['AbstractText']:
            return f"🌍 {data['AbstractText']}"
        elif 'RelatedTopics' in data and data['RelatedTopics']:
            return f"🌍 {data['RelatedTopics'][0]['Text']}"
        else:
            return "No relevant data found."
    except Exception as e:
        return f"Error fetching results: {e}"


# 🔹 AI-Generated Responses
def generate_ai_response(prompt):
    generated = generator(
        f"User: {prompt}\nAssistant:",
        max_length=80,
        truncation=True,
        num_return_sequences=1,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.5
    )[0]['generated_text']
    response = generated.split("Assistant:")[-1].strip()
    return response if response else "I didn't understand that, can you rephrase?"


# 🔹 Hybrid Response Generator
def chatbot_response(msg):
    """Decide which method to use for answering the question."""

    # ✅ Math & Science Queries (Wolfram Alpha)
    wolfram_result = ask_wolfram(msg)
    if wolfram_result:
        return f"🔢 Bot: {wolfram_result}"

    # ✅ General Knowledge (Wikipedia)
    wiki_result = search_wikipedia(msg)
    if wiki_result:
        return f"📖 Bot: {wiki_result}"

    # ✅ Real-Time Web Search (DuckDuckGo)
    if "best" in msg.lower() or "top" in msg.lower() or "hospital" in msg.lower():
        return search_duckduckgo(msg)

    # ✅ Chatbot Model Intent Responses
    ints = predict_class(msg)
    if ints:
        intent_confidence = float(ints[0][1])
        if intent_confidence > 0.65:
            matched_intent = next(
                intent for intent in intents['intents']
                if intent['tag'] == classes[ints[0][0]]
            )
            return random.choice(matched_intent['responses'])

    # ✅ AI-Generated Fallback Response
    return generate_ai_response(msg)


# 🔹 Example Usage
if __name__ == "__main__":
    print("Chatbot Ready! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Bot: {response}")
