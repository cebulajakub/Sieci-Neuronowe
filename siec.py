import numpy as np



# S - liczba wejsc sieci
# K - liczba neuronow w warstwie
# W - zwraca macierz wag sieci
def init1(S,K):

    W = np.random.uniform(low=-0.1, high=0.1, size=(K, S))
    
    return W
    
def dzialaj(W, X):
    print("Bye")       
    

if __name__ == '__main__':
    S = 3  # Liczba wejść do sieci
    K = 4  # Liczba neuronów w warstwie
    W = init1(S, K)
    print(W)