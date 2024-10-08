import numpy as np
import random
import matplotlib.pyplot as plt


# import matplotlib as plt

def sigmoid(x, beta):
    return 1 / (1 + np.exp(-beta * x))


def sigmoid_derivative(x):
    return x * (1 - x)


# S - liczba wejsc sieci
# K - liczba neuronow w warstwie
# W - zwraca macierz wag sieci
def init1(S, K):
    W = np.random.uniform(low=-0.1, high=0.1, size=(S, K))
    # W = np.random.uniform(low=-1.0, high=1.0, size=(S, K))

    return W


# W - Macierz wag sieci
# X - Wektor wejści do sieci - sygnał podany ma wejście
# Y - zwraca wektor wyjść sieci/sygnał na wyjsciu sieci
def dzialaj(W, X):
    # Mnożenie macierzowe
    # print("w",W)
    #  print(X)
    U = np.dot(W.T, X)
    beta = 1
    Y = sigmoid(U, beta)
    # print()
    return Y


# Wprzed - macierz wag przed uczeniem
# P - ciąg uczący/przykłądy/wejścia
# T - oczekiwane wyjścia
# n - liczba kroków
# Wpo - macierz wag po uczeniu
def ucz(Wprzed, P, T, krok, blad):
    W = Wprzed
    wspUcz = 0.03

    Errors_for_plot = []
    MSE = 0
    for i in range(krok):

        random_number = random.randint(0, P.shape[0] - 1)
        random_input = P[random_number, :]
        Y = dzialaj(W, random_input)
        D = T[random_number, :] - Y

        MSE += np.sum(D ** 2)
        MSE /= P.shape[0]
        Errors_for_plot.append(MSE)
        if i >= krok:
            break
        elif MSE <= blad:
            break

        dW = wspUcz * np.dot(random_input.reshape(-1, 1), D.reshape(1, -1))
        W += dW

    plt.plot(Errors_for_plot)
    plt.show()
    # print("WWWW")
    # print(Wpo)
    return W


if __name__ == '__main__':
    S = 5  # Liczba wejść do sieci
    K = 3  # Liczba neuronów w warstwie

    P = np.array([[4.0, 0.01, 0.01, -1.0, 1.5],
                  [2.0, -1.0, 2.0, 2.5, 2.0],
                  [-1.0, 3.5, 0.01, -2.0, 1.5]])

    T = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    #człowiek
    C = [2.0, 0.02, 0.02, 0.01, 0.05]
    #Nietoperz
    N = [2.0, -1, 3.0, -1.5, -1]
    Wprzed = init1(S, K)
    #Struś
    S = [2.0, 0.01, 0.01, 3.0, 2.0]
    #Wąż
    W = [0, 0.5, -1, -1, 0.5]


    print("Dane wejsciowe :")
    print("Wejscie 1 ile ma nóg")
    print("Wejscie 2 czy żyje w wodzie")
    print("Wejscie 3 czy umie latac")
    print("Wejscie 4 czy ma pióra")
    print("Wejscie 5 czy jest jajorodny")
    print("Output  ssak    ptak   ryba")
    # print(P[0, :].reshape(-1, 1))
    # Yprzed = dzialaj(Wprzed, P)
    # print("Yprzed:", Yprzed)
    krok = 100
    blad = 0.0005

    Wpo = ucz(Wprzed, P, T, krok, blad)
    for i in range(P.shape[0]):
        Ypo = dzialaj(Wpo, P[i])
        print("Input:", P[i], "Output:", Ypo)

    YpoC = dzialaj(Wpo, C)
    print("Czlowiek")
    print("Input:", C, "Output:", YpoC)
    YpoN = dzialaj(Wpo, N)
    print("Nietoperz")
    print("Input:", N, "Output:", YpoN)
    YpoS = dzialaj(Wpo, S)
    print("Sruś")
    print("Input:", S, "Output:", YpoS)
    YpoW = dzialaj(Wpo, W)
    print("Wąż")
    print("Input:", W, "Output:", YpoW)



#PROBA DODANIA MOMENTUM
    ####################################3
    # def ucz(Wprzed, P, T, krok, blad, momentum=0.9):
    # W = Wprzed
    # wspUcz = 0.03

    # Errors_for_plot = []
    # MSE = 0
    # dW_prev = np.zeros_like(W)  # Przechowywanie poprzednich zmian wag

    # for i in range(krok):
    #     random_number = random.randint(0, P.shape[0] - 1)
    #     random_input = P[random_number, :]
    #     Y = dzialaj(W, random_input)
    #     D = T[random_number, :] - Y

    #     MSE += np.sum(D ** 2)
    #     MSE /= P.shape[0]
    #     Errors_for_plot.append(MSE)

    #     if i >= krok:
    #         break
    #     elif MSE <= blad:
    #         break

    #     dW = wspUcz * np.dot(random_input.reshape(-1, 1), D.reshape(1, -1))
    #     dW = momentum * dW_prev + dW  # Uwzględnienie momentum
    #     W += dW
    #     dW_prev = dW  # Aktualizacja poprzednich zmian wag

    # plt.plot(Errors_for_plot)
    # plt.show()
    # return W