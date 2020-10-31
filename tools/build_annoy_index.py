# -*- coding: utf-8 -*-
# ----
# Build and save annoy index
# Reference: https://github.com/aparrish/plot-to-poem/blob/master/plot-to-poem.ipynb
# ----

import argparse
import pickle

import random
import spacy
from annoy import AnnoyIndex
import numpy as np

# TODO: Make it work with 'en_core_web_lg'
# Make sure to use the same model in the API!
nlp = spacy.load("en")


def main(options):

    desc = "A woman standing in front of a mirror in a room"
    input_lines = [line.rstrip("\n") for line in open(options.data, "r")]

    if options.save:

        index, lines = build_index(input_lines)
        index.save("poemIndex.ann")
        pickle.dump(lines, open("poemLines.p", "wb"))

    else:

        index = AnnoyIndex(384)
        index.load("poemIndex.ann")  # super fast, will just mmap the file
        lines = pickle.load(open("poemLines.p", "rb"))

    # TEST
    nearest = index.get_nns_by_vector(meanvector(desc), n=10)

    print("Original \n{}\nClosest meanings".format(desc))
    for n in nearest:
        print(lines[n])


def meanvector(text):
    """
  Average of the word vectors in a sentence
  """
    s = nlp(text)
    # Try better / different vector embedding
    vecs = [
        word.vector
        for word in s
        if word.pos_ in ("NOUN", "VERB", "ADJ", "ADV", "PROPN", "ADP")
        and np.any(word.vector)
    ]  # skip all-zero vectors
    if len(vecs) == 0:
        raise IndexError
    else:
        return np.array(vecs).mean(axis=0)


def build_index(data):

    t = AnnoyIndex(384, metric="angular")
    i = 0
    lines = list()
    for line in data:
        try:
            t.add_item(i, meanvector(line))
            lines.append(line)
            i += 1

        except IndexError:
            continue

    t.build(100)

    return t, lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data", type=str, help="data")
    parser.add_argument("--save", action="store_true", help="save or load")
    args = parser.parse_args()
    main(args)

