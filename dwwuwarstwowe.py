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
    #print(W1)
    return W1, W2


def dzialaj(W1, W2, X):

    beta = 1

    X_with_bias = np.append([1], X)

    Y1 = sigmoid(np.dot(W1.T, X_with_bias), beta)

    Y1_with_bias = np.append([1], Y1)

    Y2 = sigmoid(np.dot(W2.T, Y1_with_bias), beta)

    return Y1, Y2


def ucz(W1przed, W2przed, P, T, n, wspUcz, momentum):
    liczbaPrzykladow = len(P)
    W1 = W1przed
    W2 = W2przed
    dW1_prev = np.zeros_like(W1)
    dW2_prev = np.zeros_like(W2)
    Errors_for_plot = []
    indices = np.arange(liczbaPrzykladow)
    MSEPrev = 9999

    for i in range(n):
        MSE = 0  # Reset MSE for each iteration

        np.random.shuffle(indices)  # Shuffle indices for random selection
        for j in indices:
            X = P[j, :]
            X1 = np.append([1], X)
            Y1, Y2 = dzialaj(W1, W2, X)
            X2 = np.append([1], Y1)


            D2 = T[j, :] - Y2
            MSE += np.sum(D2 ** 2)


            E2 = D2 * sigmoid_derivative(Y2)
            D1 = D2 * Y1
            E1 = np.dot(W2[1:, :], E2) * sigmoid_derivative(Y1)


            dW2 = wspUcz * np.outer(X2, E2)
            dW1 = wspUcz * np.outer(X1, E1)
            dW1 += momentum * dW1_prev
            dW2 += momentum * dW2_prev
            W1 += dW1[:, :W1.shape[1]]
            W2 += dW2
            dW1_prev = dW1
            dW2_prev = dW2

        Errors_for_plot.append(MSE)
        MSE /= liczbaPrzykladow



        if MSE <= MSEPrev * 1.04:
            wspUcz *= 1.05
        elif MSE > MSEPrev * 1.04:
            wspUcz *= 0.7
        print("wspUcz: ", wspUcz)
        print(f"MSEPref:{MSEPrev},MSE {MSE}")
        MSEPrev = MSE

        #Check if MSE is below threshold
        if MSE <= 0.001:
            print(f"Wystarczy Po {i} Iteracjach")
            break

    plt.plot(Errors_for_plot)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Change in MSE during Training')
    plt.show()

    W1po = W1
    W2po = W2

    return W1po, W2po


if __name__ == '__main__':
    S = 2
    K1 = 3
    K2 = 1
    momentum = 0.9
    wspUcz = 0.5
    W1, W2 = init2(S, K1, K2)

    P = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

    T = np.array([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])

    W1po, W2po = ucz(W1, W2, P, T, 600, wspUcz, momentum)
   # Ypo1, Ypo2 = dzialaj(W1, W2, P[0])
   # print("Input:", P[0], "Output:", Ypo2)
    for i in range(P.shape[0]):
        Ypo1, Ypo2 = dzialaj(W1po, W2po, P[i])
        print("Input:", P[i], "Output:", Ypo2)


        # o Epoki czy epoka to przejszcie przez wszytkie przypadki
        # dlaczego wsp rosnie ze az strach
        # czy wykres jest odpowiedni
        #
