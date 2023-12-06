import nltk
nltk.download('averaged_perceptron_tagger')
import numpy as np
import random
import string
import re
import unicodedata
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import wikipedia as wk
from collections import defaultdict
from openfabric_pysdk.utility import SchemaUtil
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
data = open('sci.txt', 'r', errors='ignore')
raw = data.read()
raw = raw.lower()
# Sentence tokenizer
sent_tokens = nltk.sent_tokenize(raw)


# Lemmatization
def Normalize(text):
    remove_punct_dict = dict((word(punct), None) for punct in string.punctuation)
    # Word tokenization
    word_token = nltk.word_tokenize(text.lower().translate(remove_punct_dict))

    # Remove ASCII
    new_words = []
    for word in word_token:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)

    # Remove tags
    rmv = []
    for w in new_words:
        text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", w)
        rmv.append(text)

    # POS tagging and lemmatization
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in nltk.pos_tag(rmv):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list


# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(user_response):
    for word in user_response.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = linear_kernel(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0 or "tell me about" in user_response:
        print("Checking Wikipedia")
        if user_response:
            robo_response = wikipedia_data(user_response)
            return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


def wikipedia_data(input):
    reg_ex = re.search('tell me about ', input)
    try:
        if reg_ex:
            topic = reg_ex.group(1)
            ny = wk.summary(topic, sentences=1)
            return ny
    except Exception as e:
        print(e)


############################################################
# Callback function called on update config
############################################################
def config(configuration: dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        user_response = text.lower()
        if user_response != 'bye':
            if user_response == 'thanks' or user_response == 'thank you':
                output.append("Chatterbot: You are welcome..")
            else:
                greeting_resp = greeting(user_response)
                if greeting_resp is not None:
                    output.append(f"Chatterbot: {greeting_resp}")
                else:
                    response_text = response(user_response)
                    output.append(f"Chatterbot: {response_text}")
                    sent_tokens.remove(user_response)
        else:
            output.append("Chatterbot: Bye! take care..")

    return SchemaUtil.create(SimpleText(), dict(text=output))