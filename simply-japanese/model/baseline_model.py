import os
import pandas as pd
import numpy as np
import re

# Preprocessing
import MeCab
import neologdn
import collections
from nltk import FreqDist
from nltk.corpus import stopwords

import time
import logging
import collections
import logging
import time
from gensim.models.word2vec import Word2Vec
logging.basicConfig()
logging.root.setLevel(logging.INFO)

# Visualization
import matplotlib.pyplot as plt
import japanize_matplotlib

def get_data():
    """
    Gets csv data under from data
    Returns as Dataframe where columns=['original','simplified']
    """
    path = os.environ.get("LOCAL_DATA_PATH")

    if os.environ.get("DATA_SOURCE") == "TEST":
        file = os.environ.get("TEST_DATA")
    elif os.environ.get("DATA_SOURCE") == "DEPLOY":
        file = os.environ.get("DEPLOY_DATA")
    else:
        raise Exception ("Data source not or incorrectly specified in env.")

    df = pd.read_excel(os.path.join(path, file))

    df.drop(columns=['#英語(原文)','#固有名詞'], inplace=True, errors='ignore')
    df.rename(columns={"#日本語(原文)": "original", "#やさしい日本語": "simplified"}, inplace=True)

    return df


def term_frequency(df, col='original'):
    """
    Count number of terms in a corpus
    Ignore independent words  ["助動詞", "助詞", "補助記号"] and words in japanese stopwords
    Returns collection of term and its frequency
    """
    # FIXME : Need to find a way to implement japanese_stopword.txt when this file is used externally
    jp_stopwords = stopwords.words('japanese')
    all_terms = collections.Counter()
    t = MeCab.Tagger("-O wakati")
    for idx, row in df.iterrows():
        text = row[col]
        node = t.parseToNode(text).next
        while node.next:
            part_of_speech = node.feature.split(',')[0]
            # TBD
            if part_of_speech in ["助動詞", "助詞", "補助記号"] or node.surface in jp_stopwords:
                node = node.next
                continue
            all_terms[node.surface] += 1
            node = node.next
    return all_terms


def get_simplified_terms(df, n_most_common):
    """
    Only returns simplified terms that exists in the simplified column
    Return list until the top 'n' elements from most common
    """
    # Filter out corpuses if original and simplified are exactly the same
    diff_corpus_df = df[df['original'] != df['simplified']]

    # Create collections of original and simplified terms
    original_terms = term_frequency(diff_corpus_df, 'original')
    simplified_terms = term_frequency(diff_corpus_df, 'simplified')

    # Compare two collections using subtract
    diff_terms = simplified_terms
    diff_terms.subtract(original_terms)

    diff_terms_df = pd.DataFrame(dict(diff_terms).items(), columns=['word', 'count'])
    return diff_terms_df[diff_terms_df['count'] >= 0].sort_values(by='count', ascending=False)['word'].tolist()[:n_most_common]


def is_romaji(string):
    """
    returns True if string contains Romaji or a number, False if not.
    """
    alphanum_full = r'[A-z0-9]'
    if re.findall(alphanum_full, string):
        return True
    return False

def replace_terms(data, term_list, wv):
    """
    1. Identify every POS in a sentence and if it should be replaced
    2. Use the pre-trained Word2Vec model to get a term from term_list with closest distance to POS
    3. Replace POS in sentence
    4. Add new sentence to dataframe in column "prediction"

    input:
    data, np.series
    term_list, list of simplified terms
    wv, word2vec model.wv

    output:
    prediction, np.series
    """
    logging.root.setLevel(logging.INFO)

    start = time.time()
    # Make sure the data is a series, not a df or list
    try:
        assert type(data) == pd.core.series.Series
        logging.info("Data file type OK")
    except:
        print("Data file type is NOT a pd.series")

    pos_list = ("名詞", "動詞", "代名詞") # POS (part of speech) that will possibly be removed
    threshold = 0.5 # Threshold of similarity, over which a term will be replaced
    t = MeCab.Tagger()
    counter = collections.Counter()
    prediction = data.copy()
    assert len(prediction) == len(data)     # Make sure prediction and data have the same size

    # Iterate over every sentence in the dataset
    for idx, row in data.items():
        row = neologdn.normalize(row)
        logging.debug(f"Currrent sentence: {row}")
        sentence = []
        # Iterate over every word in the sentence
        node = t.parseToNode(row).next
        while node.next:
            if is_romaji(node.surface):
                sentence.append(node.surface)
                node = node.next
            word = node.feature.split(',')[8]
            part_of_speech = node.feature.split(',')[0]
            # If POS is not noun, pronoun or verb: add word to list and continue
            if part_of_speech not in pos_list:
                sentence.append(word)
            else:
                # If the term is already in the term list: do not replace, add word to list and continue
                if word in term_list:
                    sentence.append(word)
                else:
                    # Replace word with closest word from term list
                    try:
                        if wv.most_similar(word)[0][1] > threshold:
                            closest_word = wv.most_similar(word)[0][0]
                            sentence.append(closest_word)
                        else:
                            sentence.append(word)
                    except KeyError as e:
                        sentence.append(word)
                        logging.warning(f"{e}. Term will not be replaced.")
            counter[node.surface] += 1
            node = node.next
        logging.debug(sentence)
        prediction[idx] = "".join(sentence)

    assert len(data) == len(prediction)  # Make sure prediction and data have the same size
    end = time.time()
    logging.info(end-start)
    return prediction


def predict(data):
    term_list = get_simplified_terms(X150, 2000)
    model = Word2Vec.load("word2vec.gensim.model")

    replace_terms(data, term_list=term_list, wv=model.wv)
