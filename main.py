import nltk
nltk.download('averaged_perceptron_tagger')
import numpy as np
import random
import string  # to process standard python strings
import re, string, unicodedata
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import wikipedia as wk
from collections import defaultdict
nltk.download('punkt')  # first-time use only
nltk.download('wordnet')  # first-time use only

# Load the dataset
data = open('sci.txt', 'r', errors='ignore')
raw_text = data.read()
raw_text = raw_text.lower()

# Sentence tokenizer
sent_tokens = nltk.sent_tokenize(raw_text)  # Converts to list of sentences

# Lemmatization
def normalize_text(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    # Word tokenization
    word_tokens = nltk.word_tokenize(text.lower().translate(remove_punct_dict))

    # Remove ASCII
    new_words = []
    for word in word_tokens:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)

    # Remove tags
    removed = []
    for w in new_words:
        text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", w)
        removed.append(text)

    # POS tagging and lemmatization
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    removed = [i for i in removed if i]
    for token, tag in nltk.pos_tag(removed):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list

# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! You are talking to me"]

def greeting_response(user_resp):
    for word in user_resp.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

def chatbot_response(user_resp):
    robo_resp = ''
    sent_tokens.append(user_resp)
    TfidfVec = TfidfVectorizer(tokenizer=normalize_text, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = linear_kernel(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0) or "tell me about" in user_resp:
        print("Checking Wikipedia")
        if user_resp:
            robo_resp = wikipedia_search(user_resp)
            return robo_resp
    else:
        robo_resp = robo_resp + sent_tokens[idx]
        return robo_resp

# Wikipedia search
def wikipedia_search(input_text):
    reg_ex = re.search('tell me about ', input_text)
    try:
        if reg_ex:
            topic = reg_ex.group(1)
            summary = wk.summary(topic, sentences=1)
            return summary
    except Exception as e:
        print(e)

flag = True
print('''My name is Chatterbot and I'm a chatbot. If you want to exit, type end!\n
      If you want to search through Wikipedia, type "tell me about"\n
      If you want to search through documents, through document''')

while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("Chatterbot: You are welcome..")
        else:
            if greeting_response(user_response) is not None:
                print("Chatterbot: " + greeting_response(user_response))
            else:
                print("Chatterbot: ", end="")
                print(chatbot_response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("Chatterbot: Bye! Have a nice day.")
