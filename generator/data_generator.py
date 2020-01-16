import numpy as np
import scipy.special
import csv
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from generator.csv2npz import np2npz
from generator.plotting import draw_w

storage_path = '.'

def lindisc(X, t, p):
    ''' Linear MMD '''

    it = np.where(t > 0)
    ic = np.where(t < 1)

    Xc = X[ic]
    Xt = X[it]

    mean_control = np.mean(Xc, axis=0)
    mean_treated = np.mean(Xt, axis=0)

    c = np.square(2 * p - 1) * 0.25
    f = np.sign(p - 0.5)

    mmd = np.sum(np.square(p * mean_treated - (1 - p) * mean_control))
    mmd = f * (p - 0.5) + np.sqrt(c + mmd)

    return mmd


def get_multivariate_normal_params(m, dep, seed=0):
    np.random.seed(seed)

    if dep:
        mu = np.random.normal(size=m) / 10.
        ''' sample random positive semi-definite matrix for cov '''
        temp = np.random.uniform(size=(m, m))
        temp = .5 * (np.transpose(temp) + temp)
        sig = (temp + m * np.eye(m)) / 100.

    else:
        mu = np.zeros(m)
        sig = np.eye(m)

    return mu, sig


def get_latent(m, seed, n, dep):
    L = np.array((n * [[]]))
    if m != 0:
        mu, sig = get_multivariate_normal_params(m, dep, seed)
        L = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    return L


def plot(z, pi0_t1, t, y, data_path, file_name):
    gridspec.GridSpec(3, 1)

    z_min = np.min(z)  # - np.std(z)
    z_max = np.max(z)  # + np.std(z)
    z_grid = np.linspace(z_min, z_max, 100)

    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ind = np.where(t == 0)
    plt.plot(z[ind], np.squeeze(y[ind, 0]), '+', color='r')
    ind = np.where(t == 1)
    plt.plot(z[ind], np.squeeze(y[ind, 1]), '.', color='b')
    plt.legend(['t=0', 't=1'])

    ax = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    ind = np.where(t == 0)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='r', linewidth=2)
    ind = np.where(t == 1)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='b', linewidth=2)

    plt.savefig(data_path + 'info/' + file_name + '.png')
    plt.close()


def run(run_dict):
    ''' Setting... '''
    mA = run_dict["mA"]  # dimention of \Gamma
    mB = run_dict["mB"]  # dimention of \Delta
    mC = run_dict["mC"]  # dimention of \Upsilon
    mD = run_dict["mD"]  # dimention of noisy feature
    sc = run_dict["sc"]  # the logistic growth rate or steepness of the curve
    sh = run_dict["sh"]  # the x-value of the sigmoid's midpoint
    init_seed = run_dict["init_seed"]
    size = run_dict["size"]
    file = run_dict["file"]
    binary = run_dict["binary"]
    num = run_dict["num"]
    random_coef = run_dict["random_coef"]
    coef_t_AB = run_dict["coef_t_AB"]
    coef_y1_BC = run_dict["coef_y1_BC"]
    coef_y2_BC = run_dict["coef_y2_BC"]
    use_one = run_dict["use_one"]

    dep = 0  # overwright; dep=0 generates harder datasets
    n_trn = size

    seed_coef = 10
    max_dim = mA + mB + mC + mD

    which_benchmark = 'Syn_' + '_'.join(str(item) for item in [sc, sh, dep])

    temp = get_latent(max_dim, seed_coef * init_seed + 4, n_trn, dep)

    # bias same
    A = temp[:, 0:mA]
    B = temp[:, mA:mA + mB]
    C = temp[:, mA + mB:mA + mB + mC]

    D = temp[:, mA + mB + mC:mA + mB + mC + mD]

    x = np.concatenate([A, B, C, D], axis=1)
    AB = np.concatenate([A, B], axis=1)
    BC = np.concatenate([B, C], axis=1)

    # coef_t_AB
    np.random.seed(1 * seed_coef * init_seed)  # <--
    if random_coef == "True" or random_coef == "T":
        coefs_1 = np.random.normal(size=mA + mB)
    else:
        coefs_1 = np.array(coef_t_AB)
    if use_one == "True" or use_one == "T":
        coefs_1 = np.ones(shape=mA + mB)
    z = np.dot(AB, coefs_1)
    per = np.random.normal(size=n_trn)
    pi0_t1 = scipy.special.expit(sc * (z + sh + per))
    t = np.array([])
    for p in pi0_t1:
        t = np.append(t, np.random.binomial(1, p, 1))

    # coef_y_BC
    np.random.seed(2 * seed_coef * init_seed)  # <--
    if random_coef == "True" or random_coef == "T":
        coefs_2 = np.random.normal(size=mB + mC)
    else:
        coefs_2 = np.array(coef_y1_BC)
    if use_one == "True" or use_one == "T":
        coefs_2 = np.ones(shape=mB + mC)
    mu_0 = np.dot(BC ** 1, coefs_2) / (mB + mC)
    if random_coef == "True" or random_coef == "T":
        coefs_3 = np.random.normal(size=mB + mC)
    else:
        coefs_3 = np.array(coef_y2_BC)
    if use_one == "True" or use_one == "T":
        coefs_3 = np.ones(shape=mB + mC)
    mu_1 = np.dot(BC ** 2, coefs_3) / (mB + mC)

    y = np.zeros((n_trn, 2))
    if binary == "False" or binary == "F":
        np.random.seed(3 * seed_coef * init_seed)  # <--
        y[:, 0] = mu_0 + np.random.normal(loc=0., scale=.1, size=n_trn)
        np.random.seed(3 * seed_coef * init_seed)  # <--
        y[:, 1] = mu_1 + np.random.normal(loc=0., scale=.1, size=n_trn)
    else:
        mu_0 = np.dot(BC ** 1, coefs_2)
        mu_1 = np.dot(BC ** 1, coefs_3) + 1
        per = np.random.normal(size=n_trn)
        mu_0 = scipy.special.expit(sc * (mu_0 + sh + per))
        per = np.random.normal(size=n_trn)
        mu_1 = scipy.special.expit(sc * (mu_1 + sh + per))
        y_0 = np.array([])
        for p in mu_0:
            y_0 = np.append(y_0, np.random.binomial(1, p, 1))
        y_1 = np.array([])
        for p in mu_1:
            y_1 = np.append(y_1, np.random.binomial(1, p, 1))

        y[:, 0] = y_0
        y[:, 1] = y_1

    yf = np.array([])
    ycf = np.array([])
    for i, t_i in enumerate(t):
        yf = np.append(yf, y[i, int(t_i)])
        ycf = np.append(ycf, y[i, int(1 - t_i)])

    ##################################################################
    # data_path = storage_path+'/data/'+which_benchmark
    data_path = storage_path + '/data/'

    if not os.path.exists(storage_path + '/data/'):
        os.mkdir(storage_path + '/data/')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    which_dataset = '_'.join(str(item) for item in [mA, mB, mC])
    # data_path += '/'+which_dataset+'/'
    data_path = data_path + file + '_' + str(init_seed) + '_' + which_dataset + '/'
    if not os.path.exists(data_path + 'info/'):
        os.makedirs(data_path + 'info/')

    f = open(data_path + 'info/config.txt', 'w')
    f.write(which_benchmark + '_' + which_dataset)
    f.close()

    f = open(data_path + 'info/coefs.txt', 'w')
    f.write(str(coefs_1) + '\n')
    f.write(str(coefs_2) + '\n')
    f.write(str(coefs_3) + '\n')
    f.close()

    np.random.seed(4 * seed_coef * init_seed + init_seed)  # <--
    file_name = 'data'
    with open(data_path + file_name + '.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        for i in np.random.permutation(n_trn):
            temp = [t[i], yf[i], ycf[i], mu_0[i], mu_1[i]]
            temp.extend(x[i, :])
            csv_writer.writerow(temp)

    num_pts = 250
    plot(z[:num_pts], pi0_t1[:num_pts], t[:num_pts], y[:num_pts], data_path, file_name)

    with open(data_path + 'info/specs.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        temp = [np.mean(t), np.min(pi0_t1), np.max(pi0_t1), np.mean(pi0_t1), np.std(pi0_t1)]
        temp.append(lindisc(x, t, np.mean(t)))
        csv_writer.writerow(temp)


    np2npz(data_path, num)
    draw_w(coefs_1,coefs_2,coefs_3,data_path + 'info/')


