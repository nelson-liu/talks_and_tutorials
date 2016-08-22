try:
   import cPickle as pickle
except:
   import pickle
import urllib2
import numpy as np
import time


def main():
    vectors = get_vectors()
    t0 = time.time()
    analogy_solution = calculate_analogies_cython("man", "woman", "king", vectors)
    t1 = time.time()
    print t1-t0


def get_vectors():
    local_path = "../data/glove.840B.300d.pkl"
    if local_path == None:
        # download pickled word vectors
        pickled_vectors = urllib2.urlopen("http://www.nelsonliu.me/files/glove.840B.300d.pkl")
        glove_vecs = pickle.load(pickled_vectors)
    else:
        glove_vecs = pickle.load(open(local_path,"rb"))

    vocabulary = glove_vecs.keys()

    # the dictionary is {word:list}, let's make it {word:ndarray}
    # feel free to comment this out if you don't need it
    for word in vocabulary:
        glove_vecs[word] = np.array(glove_vecs[word])
    return glove_vecs

def calculate_analogies_cython(w_a, w_b, w_c, vectors):
    # get the vectors corresponding to the words
    A = vectors[w_a]
    B = vectors[w_b]
    C = vectors[w_c]

    nd_vectors = np.array(vectors.values())
    return vectors.keys()[calculate_analogies_cython_helper(w_a, w_b, w_c, A, B, C, nd_vectors)]

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.abspath(__file__)))
    from _analogies_cython import calculate_analogies_cython_helper
    main()
