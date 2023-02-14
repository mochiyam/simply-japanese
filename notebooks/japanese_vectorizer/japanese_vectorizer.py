#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import time
from multiprocessing import cpu_count
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


VECTORS_SIZE = 50
input_filename = "Input_150.xlsx"
output_filename = "model_file.txt"
output_filename_2 = "model_wv.txt"

JA_VECTORS_MODEL_FILENAME = f'ja-gensim.{VECTORS_SIZE}d.data.model'
JA_VECTORS_TEXT_FILENAME = f'ja-gensim.{VECTORS_SIZE}d.data.txt'

def generate_vectors(input_filename, output_filename, output_filename_2):
    """
    Takes a list sentences in _input_filename, and converts them


    input_filename:    list of sentences
    output_filename:   model
    output_filename_2: wv of model
    """
    if os.path.isfile(output_filename):
        logging.info('Skipping generate_vectors(). File already exists: {}'.format(output_filename))
        return

    start = time.time()

    model = Word2Vec(LineSentence(input_filename),
                     vector_size=VECTORS_SIZE,
                     window=5,
                     min_count=5,
                     workers=4,
                     epochs=5)

    model.save(output_filename)
    model.wv.save_word2vec_format(output_filename_2, binary=False)

    logging.info('Finished generate_vectors(). It took {0:.2f} s to execute.'.format(round(time.time() - start, 2)))


def get_words(text):

    import MeCab
    mt = MeCab.Tagger('-d /usr/lib/mecab/dic/mecab-ipadic-neologd')

    mt.parse('')

    parsed = mt.parseToNode(text)
    components = []

    while parsed:
        components.append(parsed.surface)
        parsed = parsed.next

    return components


def tokenize_text(input_filename, output_filename):

    if os.path.isfile(output_filename):
        logging.info('Skipping tokenize_text(). File already exists: {}'.format(output_filename))
        return

    start = time.time()

    with open(output_filename, 'w') as out:
        with open(input_filename, 'r') as inp:

            for i, text in enumerate(inp.readlines()):

                tokenized_text = ' '.join(get_words(text))

                out.write(tokenized_text + '\n')

                if i % 100 == 0 and i != 0:
                    logging.info('Tokenized {} articles.'.format(i))
    logging.info('Finished tokenize_text(). It took {0:.2f} s to execute.'.format(round(time.time() - start, 2)))


def process_wiki_to_text(input_filename, output_text_filename, output_sentences_filename):

    if os.path.isfile(output_text_filename) and os.path.isfile(output_sentences_filename):
        logging.info('Skipping process_wiki_to_text(). Files already exist: {} {}'.format(output_text_filename,
                                                                                          output_sentences_filename))
        return

    start = time.time()
    intermediary_time = None
    sentences_count = 0

    with open(output_text_filename, 'w') as out:
        with open(output_sentences_filename, 'w') as out_sentences:

            # Open the Wiki Dump with gensim
            wiki = WikiCorpus(input_filename, lemmatize=False, dictionary={}, processes=cpu_count())
            wiki.metadata = True
            texts = wiki.get_texts()

            for i, article in enumerate(texts):
                # article[1] refers to the name of the article.
                text_list = article[0]
                sentences = text_list
                sentences_count += len(sentences)

                # Write sentences per line
                for sentence in sentences:
                    out_sentences.write((sentence + '\n'))

                # Write each page in one line
                text = ' '.join(sentences) + '\n'
                out.write(text)

                # This is just for the logging
                if i % (100 - 1) == 0 and i != 0:
                    if intermediary_time is None:
                        intermediary_time = time.time()
                        elapsed = intermediary_time - start
                    else:
                        new_time = time.time()
                        elapsed = new_time - intermediary_time
                        intermediary_time = new_time
                    sentences_per_sec = int(len(sentences) / elapsed)
                    logging.info('Saved {0} articles containing {1} sentences ({2} sentences/sec).'.format(i + 1,
                                                                                                           sentences_count,
                                                                                                           sentences_per_sec))
        logging.info(
            'Finished process_wiki_to_text(). It took {0:.2f} s to execute.'.format(round(time.time() - start, 2)))

if __name__ == '__main__':
    print(os.getcwd())
