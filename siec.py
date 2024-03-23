import numpy as np



# S - liczba wejsc sieci
# K - liczba neuronow w warstwie
# W - zwraca macierz wag sieci
def init1(S,K):

    W = np.random.uniform(low=-0.1, high=0.1, size=(K, S))
    
    return W
    
def dzialaj(W, X):
    print("Bye")       