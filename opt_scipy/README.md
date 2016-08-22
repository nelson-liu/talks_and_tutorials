# Optimizing Scientific Python

In this tutorial, I'll go over a variety of ways to make your data science /
machine learning research code fast. I'll start by writing a basic Python
implementation of cosine similarity between two vectors, demonstrate the
performance benefits of list comprehensions and numpy, and try out some
implementations included in scipy and scikit-learn. Then, we'll go over JITs and
Numba, and show that it can lead to significant performance boosts with minimal
effort. Lastly, we'll take a look at Cython, the purpose of the GIL, and how to
easily use multithreading with Cython.
