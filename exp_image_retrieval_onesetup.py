# ploting all the results for one image retrieval setup, including all the k values
# only plots the results for L1, OT, and OTP-metric(alpha)(with different k values)

import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--delta', type=float, default=0.001)
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--shift', type=int, default=150)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--noise_type', type=str, default='whiteout')
    parser.add_argument('--x_axis', type=str, default='shift')
    parser.add_argument('--shift_st', type=int, default=0)
    parser.add_argument('--shift_ed', type=int, default=160)
    parser.add_argument('--shift_d', type=int, default=10)
    args = parser.parse_args()
    print(args)

    n = int(args.n)
    delta = args.delta
    data_name = args.data_name
    noise = args.noise
    shift = args.shift
    top_k = args.top_k
    verbose = args.verbose
    noise_type = args.noise_type
    x_axis = args.x_axis

    metric_scaler = [1.0, 10.0, 100.0]
    shift_range = np.arange(args.shift_st, args.shift_ed, args.shift_d)
    noise_range = np.arange(0.0, 1.1, 0.1)

#######collect data for w1 metric#########
    path = "./results/1was/50_500"
    OTP_metric_res_w1 = []
    OTP_metric_labels = []
    L1_res = []
    OT_res = []
    OTP_metric_res_w1.append([])
    L1_res.append([])
    OT_res.append([])
    for metric_scaler_i in metric_scaler:
        file_name = "{}/img_retrival_res_n_{}_delta_{}_data_{}_noise_{}_ms_{}_sp_{}_nt_{}_noise_vs_acc.csv".format(path, n, 0.01, data_name, noise, metric_scaler_i, shift, noise_type)
        try:
            cur_res = np.loadtxt(file_name, delimiter=",", dtype=float)
            OTP_metric_res_w1[-1].append(cur_res[:,1])
            L1_res[-1].append(cur_res[:,0])
            OT_res[-1].append(cur_res[:,-1])
            OTP_metric_labels.append("ours, w1, k={}, delta={}".format(1/metric_scaler_i, 0.01))
        except:
            print("file {} not found".format(file_name))
    OTP_metric_res_w1 = np.array(OTP_metric_res_w1)
    OTP_metric_res_w1 = OTP_metric_res_w1.squeeze()

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if x_axis == 'noise':
        x = noise_range
        x_label = 'noise'
    else:
        x = shift_range
        x_label = 'shift'

    for OTP_metric_res_w1_i, OTP_metric_label_i in zip(OTP_metric_res_w1, OTP_metric_labels):
        ax.plot(x, OTP_metric_res_w1_i.flatten(), label=OTP_metric_label_i)

#######collect data for w2 metric#########
    path = "./results/2was/50_500/precise_noise"
    OTP_metric_res_w2 = []
    OTP_metric_labels_w2 = []
    OTP_metric_res_w2.append([])
    for metric_scaler_i in metric_scaler:
        file_name = "{}/img_retrival_res_n_{}_delta_{}_data_{}_noise_{}_ms_{}_sp_{}_nt_{}_noise_vs_acc.csv".format(path, n, 0.001, data_name, noise, metric_scaler_i, shift, noise_type)
        try:
            cur_res = np.loadtxt(file_name, delimiter=",", dtype=float)
            OTP_metric_res_w2[-1].append(cur_res[:,1])
            L1_res[-1].append(cur_res[:,0])
            OT_res[-1].append(cur_res[:,-1])
            OTP_metric_labels_w2.append("ours, w2, k={}, delta={})".format(1/metric_scaler_i, 0.001))
        except:
            print("file {} not found".format(file_name))
    OTP_metric_res_w2 = np.array(OTP_metric_res_w2)
    OTP_metric_res_w2 = OTP_metric_res_w2.squeeze()
    L1_res = np.array(L1_res)
    L1_res = L1_res.squeeze()
    OT_res = np.array(OT_res)
    OT_res = OT_res.squeeze()

    # plot the results
    fig = plt.figure()
    # make a larger plot leave some space for legend on the right
    fig.set_size_inches(10, 5)
    # put figure on the left
    ax = fig.add_subplot(121)
    if x_axis == 'noise':
        x = noise_range
        x_label = 'noise'
    else:
        x = shift_range
        x_label = 'shift'

    line_styles = ['solid', 'dotted', 'dashed', 'dashdot', (0,(1,10)), (0,(5,10))]
    color = ['r', 'g', 'b', 'k']
    for OTP_metric_res_w1_i, OTP_metric_label_i, ls in zip(OTP_metric_res_w1, OTP_metric_labels, line_styles):
        ax.plot(x, OTP_metric_res_w1_i.flatten(), label=OTP_metric_label_i, linestyle=ls, color='r')

    for OTP_metric_res_w2_i, OTP_metric_label_i, ls in zip(OTP_metric_res_w2, OTP_metric_labels_w2, line_styles):
        ax.plot(x, OTP_metric_res_w2_i.flatten(), label=OTP_metric_label_i, linestyle=ls, color='b')
    
    ax.plot(x, L1_res[0].flatten(), label="L1", color='g')
    ax.plot(x, OT_res[0].flatten(), label="OT(w1)", color='black', linestyle='-')
    ax.plot(x, OT_res[3].flatten(), label="OT(w2)", color='black', linestyle='--')
    ax.legend()
    ax.legend(bbox_to_anchor=(1.1,1), loc="upper left")
    ax.set_xlabel(x_label)
    ax.set_ylabel("accuracy")
    # title shold include delta values
    title = "image retrieval results, n={}, data={}, delta={}".format(n, data_name, delta)
    ax.set_title(title)
    plt.show()

    # save the plt
    file_name = "{}/img_retrival_res_n_{}_delta_{}_data_{}_noise_{}_ms_{}_sp_{}_nt_{}_noise_vs_acc.png".format(path, n, delta, data_name, noise, metric_scaler, shift, noise_type)
    fig.savefig(file_name)
