import numpy as np
from scipy.linalg import lu_factor, lu_solve, sqrtm
from scipy.integrate import dblquad
from scipy.special import erf
from tqdm import tqdm

def teacher(h_star, symmetry):   
    if symmetry == "TEST":
        return np.tanh(h_star[0] + h_star[1]**3 - 3*h_star[1])
    else:
        raise NotImplementedError


def student(h, a):
    # ReLU student
    return a @ np.maximum(0, h) 

def loss(a, h, h_star, symmetry):
    y_teacher = teacher(h_star, symmetry)
    y_student = student(h, a)

    return (y_student - y_teacher)**2 / 2


def d_loss(a, h, h_star, symmetry):
    y_teacher = teacher(h_star, symmetry)
    y_student = student(h, a)

    return np.einsum("s,p,ps->ps", y_student - y_teacher, a, (h > 0).astype(float))


def dd_loss(a, h, h_star, symmetry):
    der = np.einsum("p,ps->ps", a, h > 0) 
    return np.einsum("ps,qs->pqs", der, der)


def kroneker_delta(x,y):
    if x == y:
        return 1
    else:
        return 0
    

def update_h(a, h_prev, lambda_eff, h_star, m_last, memory_kernel, eta, noise, symmetry):
    loss_part = d_loss(a, h_prev[-1] + m_last@h_star, h_star, symmetry)
    memory_part = np.einsum("tpq,tqs->ps", memory_kernel, h_prev[:-1])
    regularisation_part = np.einsum("pq,qs->ps", lambda_eff, h_prev[-1])

    return h_prev[-1] + eta * (- loss_part - regularisation_part + memory_part + noise)


def update_m(m_prev, grad_magnetisation, eta, lambd):
    return m_prev - eta * m_prev * lambd - grad_magnetisation * eta


def update_memory(t, tau, memory_process, lambda_eff, a, h_last, h_star, m_last, memory_kernel, eta, symmetry):
    temp_loss = memory_process[t,tau] - kroneker_delta(t,tau) 
    loss_part = np.einsum("pqs,qrs->prs", dd_loss(a, h_last + m_last@h_star, h_star, symmetry), temp_loss)
    regularisation_part = lambda_eff @ memory_process[t,tau]


    if t > 1:
        memory_part = np.einsum("tpq,tqrs->prs", memory_kernel[t,tau:t-1], memory_process[tau:t-1,tau]) 
    else:
        memory_part = 0
        
    return memory_process[t,tau] + eta * (- regularisation_part - loss_part + memory_part)


def find_correlation(alpha, a, h_t, h_tau, h_star, m_t, m_tau, symmetry):
    return alpha * np.einsum("ps,qs->pq", d_loss(a, h_t + m_t@h_star, h_star, symmetry), d_loss(a, h_tau + m_tau@h_star, h_star, symmetry)) / h_t.shape[-1]


def find_regularisation(lambd, alpha, a, h_t, h_star, m_t, symmetry):
    # I put a diagonal regularisation, check this
    return alpha * np.mean(dd_loss(a, h_t + m_t@h_star, h_star, symmetry), axis=-1) + lambd * np.eye(h_t.shape[0])


def find_grad_magnetisation(alpha, a, h_t, h_star, m_t, symmetry):
    return alpha * np.einsum("ks,ps->pk", h_star, d_loss(a, h_t + m_t@h_star, h_star, symmetry)) / h_t.shape[-1]


def find_memory_kernel(t, tau, alpha, memory_process, a, h_t, h_star, m_t, symmetry):
    return alpha * np.einsum("pws,wqs->pq", memory_process[t,tau], dd_loss(a, h_t + m_t@h_star, h_star, symmetry)) / h_t.shape[-1]


def initial_conditions(alpha, eta, lambd, a, h_0, h_star, P, K, symmetry):
    m_0 = np.zeros((P,K))
    C_0_0 = find_correlation(alpha, a, h_0, h_0, h_star, m_0, m_0, symmetry)
    grad_magnetisation_0 = find_grad_magnetisation(alpha, a, h_0, h_star, m_0, symmetry)
    lambda_eff_0 = find_regularisation(lambd, alpha, a, h_0, h_star, m_0, symmetry)
    m_1 = - grad_magnetisation_0*eta

    return m_1, C_0_0, lambda_eff_0, grad_magnetisation_0


def next_noise_step(C, samples):

    T,_ ,P,_ = C.shape
    C = np.transpose(C, (0, 2, 1, 3)).reshape(T*P, T*P)
    omega_fold = sqrtm(C) @ np.random.normal(0,1, (T*P, samples))
    return omega_fold.reshape(T,P,samples)[-1,:,:]



def DMFT(alpha, eta, lambd, T, K, P, symmetry, samples):
    a = np.ones(P) / np.sqrt(P)
    a[1::2] = -1 / np.sqrt(P)

    h_0 = np.random.normal(0,1, (P,samples))
    h_star = np.random.normal(0,1, (K,samples))

    m_1, C_0_0, lambda_eff_0, grad_magnetisation_0 = initial_conditions(alpha, eta, lambd, a, h_0, h_star, P, K, symmetry)

    h = np.zeros((T, P, samples))
    h[0] = h_0

    m = np.zeros((T+1, P,K))
    m[1] = m_1

    C = np.zeros((T,T, P,P))
    C[0,0] = C_0_0

    lambda_eff = np.zeros((T, P,P))
    lambda_eff[0] = lambda_eff_0

    memory_process = np.zeros((T,T, P,P, samples))

    R = np.zeros((T,T, P,P))

    grad_magnetisation = np.zeros((T, P,K))
    grad_magnetisation[0] = grad_magnetisation_0

    noise = np.zeros((T, P, samples))
    noise[0] = next_noise_step(C[:1, :1], samples)


    # END OF INITIALISATION

    for t in tqdm(range(1,T)):
        
        h[t] = update_h(a, h[:t], lambda_eff[t-1], h_star, m[t-1], R[t-1, :t-1], eta, noise[t-1], symmetry)

        grad_magnetisation[t] = find_grad_magnetisation(alpha, a, h[t], h_star, m[t], symmetry)
        m[t+1] = update_m(m[t], grad_magnetisation[t], eta, lambd)

        for tau in range(t):
            C[t, tau] = find_correlation(alpha, a, h[t], h[tau], h_star, m[t], m[tau], symmetry)
            C[t, tau] = (C[t, tau] + C[t, tau].T) / 2
            C[tau, t] = C[t, tau]
        C[t, t] = find_correlation(alpha, a, h[t], h[t], h_star, m[t], m[t], symmetry)
        C[t, t] = (C[t, t] + C[t, t].T) / 2


        lambda_eff[t] = find_regularisation(lambd, alpha, a, h[t], h_star, m[t], symmetry)

        for tau in range(t):
            memory_process[t, tau] = update_memory(t-1, tau, memory_process, lambda_eff[t-1], a, h[t-1], h_star, m[t-1], R[:t,:t-1], eta, symmetry)
            R[t,tau] = find_memory_kernel(t, tau, alpha, memory_process, a, h[t], h_star, m[t], symmetry)

            
        noise[t] = next_noise_step(C[:t+1,:t+1], samples)
    return m, C, lambda_eff, R, grad_magnetisation


def main():
    alpha = 5
    T = 7

    # teacher hidden layer size
    K = 2
    
    # student hidden layer size
    P = 2

    lambd = .0
    eta = .2

    samples = int(5e6)

    symmetry="TEST"

    m, C, lambda_eff, R, grad_magnetisation = DMFT(alpha, eta, lambd, T, K, P, symmetry, samples)


    # We don't save the last magnetisation point out of convenience
    np.savez(f'data/DMFT_A{alpha}_T{T}_P{P}_sym={symmetry}_lambda{lambd}_lr{eta}_samples{samples}.npz', m=m[:-1], C=C, lambda_eff=lambda_eff, R=R, grad_magnetisation=grad_magnetisation)


if __name__ == "__main__":
    main()