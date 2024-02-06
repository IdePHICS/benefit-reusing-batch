import numpy as np
from tqdm import tqdm
import argparse
from scipy.linalg import orth

import torch
from torch import nn
from torch.nn.functional import relu


class NeuralNetworkStudent(nn.Module):
    def __init__(self, D,K, start_weight_student):
        super().__init__()
        self.D = D
        self.K = K
        self.first_layer = nn.Linear(D,K, bias=False)

        self.first_layer.weight.data = torch.tensor(start_weight_student, dtype=torch.float32, requires_grad=True)
        

        with torch.no_grad():
            self.a = torch.ones(K, requires_grad=False) / np.sqrt(self.K)
            self.a[1::2] = -1 / np.sqrt(self.K)

    def forward(self, x):
        x = self.first_layer(x) / np.sqrt(self.D)
        x = relu(x) @ self.a 
        return x

class NeuralNetworkTeacher(nn.Module):
    def __init__(self, D,K, target, start_weight_teacher):
        super().__init__()
        self.target = target
        self.D = D
        self.K = K
        self.first_layer = nn.Linear(D,K, bias=False)

        self.first_layer.weight.data = torch.tensor(start_weight_teacher, dtype=torch.float32, requires_grad=False)


    def forward(self, x):
        x = self.first_layer(x) / np.sqrt(self.D)
        if self.target == 'TEST':
            return torch.tanh(x[:,0] + x[:,1]**3 - 3 * x[:,1])
        else:
            raise ValueError("target not implemented yet.")

def loss_func(y_true, y_model):
    return torch.sum((y_true-y_model)**2)/2


def set_weight(p, k, D, m_0: np.array):
    assert(m_0.shape == (p,k))
    norm = lambda w: np.linalg.norm(w, ord=2, axis=1, keepdims=True)

    weight_teacher = np.random.normal(0,1, (k,D))
    weight_teacher = weight_teacher / norm(weight_teacher)
    weight_teacher = orth(weight_teacher.T).T * np.sqrt(D)

    Wtild = np.random.normal(size=(p,D)) 
    Wtild = Wtild / norm(Wtild) * np.sqrt(D)
    Wtild_over_Wtarget = np.einsum('ji,ri,rh->jh', Wtild , weight_teacher ,weight_teacher) / D

    Worth =  Wtild - Wtild_over_Wtarget
    Worth = Worth / norm(Worth) 
    Worth = orth(Worth.T).T * np.sqrt(D)

    W0 = m_0 @ weight_teacher + np.einsum('j,ji->ji',np.sqrt(1-np.linalg.norm(m_0,ord=2,axis=1)),Worth)

    return weight_teacher, W0

def main(D, alpha, P, K, target, lr=.2, T=7, reps=32):
    N = int(alpha*D)
    magnetisation = np.zeros((T, P, K, reps)) 
    gen_error = np.zeros((T, reps))

    for rep in range(reps):
        with torch.no_grad():
            X_train = torch.normal(0,1,[N,D])
            X_test = torch.normal(0,1,[N,D])

            start_weight_teacher, start_weight_student = set_weight(P, K, D, np.zeros((P,K)))

            net_teacher = NeuralNetworkTeacher(D,P, target, start_weight_teacher)
            y_train = net_teacher(X_train)
            y_test = net_teacher(X_test)

        net_student = NeuralNetworkStudent(D,K, start_weight_student)
        optimiser = torch.optim.SGD(net_student.parameters(), lr=lr, weight_decay=0)

        for t in range(T):
            with torch.no_grad():
                magnetisation[t, :, :, rep] = net_student.first_layer.weight.data.numpy()@net_teacher.first_layer.weight.data.numpy().T / D
                gen_error[t, rep] = loss_func(y_test, net_student(X_test)).item() / N

                print(f"rep: {rep}, t: {t}, gen_error: {gen_error[t, rep]}")

            optimiser.zero_grad()

            y_model = net_student(X_train)
            loss = loss_func(y_train, y_model)

            loss.backward()
            optimiser.step()


    # save grad_magnetisation
    np.savez(f'data/simulations_A{alpha}_p{P}_k{K}_D{D}_T{T}_sym={target}_lambda{0.0}_lr{lr}_online{False}.npz', magnetisation=magnetisation, gen_error=gen_error)


if __name__ == '__main__':
    # # Parameters using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--D', type=int, default=10000)
    parser.add_argument('--alpha', type=float, default=5)
    parser.add_argument('--P', type=int, default=2)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--target', type=str, default='TEST')
    args = parser.parse_args()
    D = args.D
    alpha = args.alpha
    P = args.P
    K = args.K
    target = args.target

    # print parameters
    print(f'D: {D}')
    print(f'alpha: {alpha}')
    print(f'P: {P}')
    print(f'K: {K}')
    print(f'target: {target}')

    main(D, alpha, P, K, target)