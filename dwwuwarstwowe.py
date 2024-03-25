import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x, beta):
    return 1 / (1 + np.exp(-beta *x))


def sigmoid_derivative(x):
    return x * (1 - x)


def init2(S, K1, K2):
    W1 = np.random.uniform(low=-0.1, high=0.1, size=(S + 1, K1))
    W1[0, :] = -1  # bias
    W2 = np.random.uniform(low=-0.1, high=0.1, size=(K1 + 1, K2))
    W2[0, :] = -1  # bias
    return W1, W2


def dzialaj(W1, W2, X):

    beta = 1

    X_with_bias = np.append([1], X)

    Y1 = sigmoid(np.dot(W1.T, X_with_bias), beta)

    Y1_with_bias = np.append([1], Y1)

    Y2 = sigmoid(np.dot(W2.T, Y1_with_bias), beta)

    return Y1, Y2


def ucz(W1przed, W2przed, P, T, n):
    liczbaPrzykladow = len(P)
    W1 = W1przed
    W2 = W2przed
    S2 = W2.shape[0]
    wspUcz = 0.03


    for i in range(n):
        for example in range(liczbaPrzykladow):
            los = np.random.randint(liczbaPrzykladow)
            nrPrzykladu = los
            X = P[nrPrzykladu, :]
            X1 = np.append([1], X)
            Y1, Y2 = dzialaj(W1, W2, X)
            X2 = np.append([1], Y1)

            # blad w warstwie przy wyjsciu
            D2 = T[nrPrzykladu, :] - Y2
            E2 = D2 * sigmoid_derivative(Y2)

            # backprop
            E1 = np.dot(W2[1:, :], E2) * sigmoid_derivative(Y1)


            dW2 = wspUcz * np.outer(X2, E2)
            dW1 = wspUcz * np.outer(X1, E1)


            W1 += dW1[:, :W1.shape[1]]
            W2 += dW2

    W1po = W1
    W2po = W2

    return W1po, W2po


if __name__ == '__main__':
    S = 2
    K1 = 2
    K2 = 1
    W1, W2 = init2(S, K1, K2)

    P = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

    T = np.array([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])

    W1po, W2po = ucz(W1, W2, P, T, 100000)

    for i in range(P.shape[0]):
        Ypo1, Ypo2 = dzialaj(W1po, W2po, P[i])
        print("Input:", P[i], "Output:", Ypo2)
