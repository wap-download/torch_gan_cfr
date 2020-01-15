import numpy as np
import csv
import random

def csv2np(datafile):
    with open(datafile, "r") as csvfile:
        reader = csv.reader(csvfile)
        # 这里不需要readlines
        arr = []
        for line in reader:
            arr.append(line)

    arr = np.array(arr)
    arr = arr[np.argsort(arr[:, 0])]

    ate = np.mean(arr[:, 4:5].astype(np.float)) - np.mean(arr[:, 3:4].astype(np.float))
    n = arr.shape[0]

    n1 = int(np.sum(arr.astype(float)[:, 0]))
    n0 = n - n1

    I_0 = np.random.permutation(range(0, n0))
    I_1 = np.random.permutation(range(n0, n))

    I_0_tvt = [I_0[0:round(n0 * 0.63)], I_0[round(n0 * 0.63):round(n0 * 0.90)], I_0[round(n0 * 0.90):n0]]
    I_1_tvt = [I_1[0:round(n1 * 0.63)], I_1[round(n1 * 0.63):round(n1 * 0.90)], I_1[round(n1 * 0.90):n1]]

    I_train = list(I_0_tvt[0]) + list(I_1_tvt[0])
    I_valid = list(I_0_tvt[1]) + list(I_1_tvt[1])
    I_test = list(I_0_tvt[2]) + list(I_1_tvt[2])

    train = arr[I_train, :].astype(float)
    valid = arr[I_valid, :].astype(float)
    test = arr[I_test, :].astype(float)

    I_train = list(range(0, len(train)))
    I_valid = list(range(len(train), len(train) + len(valid)))
    n_ = [len(train), len(train) - int(np.sum(train[:, 0])), int(np.sum(train[:, 0]))]
    train = np.concatenate((train, valid), axis=0)
    test = test
    return train, I_train, I_valid, n_, test, ate

def np2npz(file_path, num = 10):
    datafile = file_path + "data.csv"

    xs = []
    ts = []
    yfs = []
    ycfs = []
    mu0s = []
    mu1s = []
    es = []
    n = []
    I_trains = []
    I_valids = []

    t_xs = []
    t_ts = []
    t_yfs = []
    t_ycfs = []
    t_mu0s = []
    t_mu1s = []
    t_es = []
    for i in range(num):
        train, I_train, I_valid, n_, test, ate= csv2np(datafile = datafile)

        x = train[:, 5:].astype(np.float)
        x = np.expand_dims(x, axis=2)
        t = train[:, 0:1].astype(np.float)
        e = t + 1. - t
        yf = train[:, 1:2].astype(np.float)
        ycf = train[:, 2:3].astype(np.float)
        mu0 = train[:, 3:4].astype(np.float)
        mu1 = train[:, 4:5].astype(np.float)

        xs.append(x)
        ts.append(t)
        es.append(e)
        yfs.append(yf)
        I_trains.append(I_train)
        I_valids.append(I_valid)
        n.append(n_)
        ycfs.append(ycf)
        mu0s.append(mu0)
        mu1s.append(mu1)


        t_x = test[:, 5:].astype(np.float)
        t_x = np.expand_dims(t_x, axis=2)
        t_t = test[:, 0:1].astype(np.float)
        t_e = t_t + 1. - t_t
        t_yf = test[:, 1:2].astype(np.float)
        t_ycf = test[:, 2:3].astype(np.float)
        t_mu0 = test[:, 3:4].astype(np.float)
        t_mu1 = test[:, 4:5].astype(np.float)

        t_xs.append(t_x)
        t_ts.append(t_t)
        t_es.append(t_e)
        t_yfs.append(t_yf)
        t_ycfs.append(t_ycf)
        t_mu0s.append(t_mu0)
        t_mu1s.append(t_mu1)

    n = np.array(n)
    xs = np.squeeze(np.array(xs))
    ts = np.squeeze(np.array(ts))
    es = np.squeeze(np.array(es))
    yfs = np.squeeze(np.array(yfs))
    ycfs = np.squeeze(np.array(ycfs))
    mu0s = np.squeeze(np.array(mu0s))
    mu1s = np.squeeze(np.array(mu1s))

    xs = np.swapaxes(xs, 0, 1)
    xs = np.swapaxes(xs, 1, 2)
    ts = np.swapaxes(ts, 0, 1)
    es = np.swapaxes(es, 0, 1)
    yfs = np.swapaxes(yfs, 0, 1)
    ycfs = np.swapaxes(ycfs, 0, 1)
    mu0s = np.swapaxes(mu0s, 0, 1)
    mu1s = np.swapaxes(mu1s, 0, 1)

    t_xs = np.squeeze(np.array(t_xs))
    t_ts = np.squeeze(np.array(t_ts))
    t_es = np.squeeze(np.array(t_es))
    t_yfs = np.squeeze(np.array(t_yfs))
    t_ycfs = np.squeeze(np.array(t_ycfs))
    t_mu0s = np.squeeze(np.array(t_mu0s))
    t_mu1s = np.squeeze(np.array(t_mu1s))

    t_xs = np.swapaxes(t_xs, 0, 1)
    t_xs = np.swapaxes(t_xs, 1, 2)
    t_ts = np.swapaxes(t_ts, 0, 1)
    t_es = np.swapaxes(t_es, 0, 1)
    t_yfs = np.swapaxes(t_yfs, 0, 1)
    t_ycfs = np.swapaxes(t_ycfs, 0, 1)
    t_mu0s = np.swapaxes(t_mu0s, 0, 1)
    t_mu1s = np.swapaxes(t_mu1s, 0, 1)

    np.savez(file_path + 'Syn.train.npz', mu1=mu1s, mu0=mu0s, yf=yfs, ycf=ycfs, t=ts, x=xs, ate=ate, e=es, n=n,
             I_trains=I_trains, I_valids=I_valids)
    np.savez(file_path + 'Syn.test.npz', mu1=t_mu1s, mu0=t_mu0s, yf=t_yfs, ycf=t_ycfs, t=t_ts, x=t_xs,
             ate=ate, e=t_es)
