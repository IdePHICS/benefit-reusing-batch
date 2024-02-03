import numpy as np
from tqdm import tqdm
from scipy.linalg import orth

def net_teacher(w, x):
    hstar = w@x.T / np.sqrt(w.shape[1])
    return np.tanh(hstar[0] + hstar[1]**3 - 3*hstar[1])


def net_student(a, w, x):
    h = w@x.T / np.sqrt(w.shape[1])
    return np.einsum("p,pn->n", a, np.maximum(h,0))

def net_student_derivative(a, w, x):
    h = w@x.T / np.sqrt(w.shape[1])
    return np.einsum("p,pn->pn", a, np.heaviside(h, 1))


def grad_w(a, weight_student, x, y):
    p = weight_student.shape[0]
    N = x.shape[0]
    
    displacements = y - net_student(a, weight_student,x)
    assert(displacements.shape == (N,))

    Displacements = np.tile(displacements, (p,1))
    assert(Displacements.shape == (p,N))

    return -Displacements*net_student_derivative(a, weight_student,x) @ x / np.sqrt(weight_student.shape[1])


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


def committee(p, k, D, N, T, m_0, lr, lambd, online=False):
    a = np.ones(p) / np.sqrt(p)
    a[1::2] = -1 / np.sqrt(p)

    weight_teacher, weight_student = set_weight(p, k, D, m_0)

    x_train = np.random.normal(0,1,[N,D])
    x_test = np.random.normal(0,1,[N,D])

    y_train = net_teacher(weight_teacher, x_train)
    y_test = net_teacher(weight_teacher, x_test)


    magnetisation = np.zeros((T, p, k))
    norm = np.zeros((T, p, p))
    gen_error = np.zeros(T)

    for i in range(T):
        magnetisation[i] = weight_student @ weight_teacher.T / D
        norm[i] = weight_student @ weight_student.T / D

        y_pred = net_student(a, weight_student, x_test)
        gen_error[i] = np.sum((y_test - y_pred)**2/2) / N

        # print('P=', weight_teacher @ weight_teacher.T / D)
        # print('M=',magnetisation[i])
        # print('  ', np.linalg.norm(magnetisation[i], ord=2, axis=1))
        # print('Q=', norm[i])
        # print('R=',gen_error[i])#, weight_student @ weight_teacher.T / np.linalg.norm(weight_student)/np.linalg.norm(weight_teacher))

        weight_student = weight_student - lr * grad_w(a, weight_student, x_train, y_train)

        if online:
            x_train = np.random.normal(0,1,[N,D])
            y_train = net_teacher(weight_teacher, x_train)
    return magnetisation, norm, gen_error

def main():
    alpha = 5
    D = 10000
    N = int(alpha*D)
    T_sim = 7
    lambd = 0.0
    lr = .2
    p = 2
    k = 2
    m_0 = np.zeros((p,k))
    symmetry_string = "TEST"
    
    samples_sim = 32

    online = False

    magnetisation_list = np.zeros((samples_sim,T_sim,p,k))
    norm_list = np.zeros((samples_sim,T_sim,p,p))
    gen_error_list = np.zeros((samples_sim,T_sim))
    for i in tqdm(range(samples_sim)):
        magnetisation_list[i], norm_list[i], gen_error_list[i] = committee(p,k,D, N, T_sim, m_0, lr, lambd, online)

    # m_0_string = str(m_0).replace('\n','').replace(' ','')
    if online:
        print('online')
        np.savez(f'data/simulations_A{alpha}_p{p}_k{k}_D{D}_T{T_sim}_sym={symmetry_string}_lambda{lambd}_lr{lr}_online{online}.npz', magnetisation_list=magnetisation_list, norm_list=norm_list, gen_error_list=gen_error_list)
    else:
        print('batch')
        np.savez(f'data/simulations_A{alpha}_p{p}_k{k}_D{D}_T{T_sim}_sym={symmetry_string}_lambda{lambd}_lr{lr}_online{online}.npz', magnetisation_list=magnetisation_list, norm_list=norm_list, gen_error_list=gen_error_list)

if __name__=="__main__":
    main()