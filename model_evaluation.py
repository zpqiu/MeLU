import os
import torch
import pickle
from torch.nn import functional as F
import numpy as np

from MeLU import MeLU
from options import config, states



def evaluate(melu, total_dataset):
    if config['use_cuda']:
        melu.cuda()

    training_set_size = len(total_dataset)
    melu.eval()

    loss_all = list()
    a,b,c,d = zip(*total_dataset)
    for i in range(training_set_size):
        try:
            supp_x = a[i]
            supp_y = b[i]
            query_x = c[i]
            query_y = d[i]
        except IndexError:
            continue

        if config['use_cuda']:
            supp_x = supp_x.cuda()
            supp_y = supp_y.cuda()
            query_x = query_x.cuda()

        _, query_set_y_pred = melu(supp_x, supp_y, query_x, 1) # eval时 local update step写死为 1
        loss_all.append(F.l1_loss(query_y, query_set_y_pred.squeeze().detach().cpu()))
    
    loss_all = np.array(loss_all)
    print('{}+/-{}'.format(np.mean(loss_all), 1.96*np.std(loss_all,0)/np.sqrt(len(loss_all))))


if __name__ == "__main__":
    master_path= "./ml"

    # training model.
    melu = MeLU(config)
    melu.cpu()
    model_filename = "{}/models.pkl".format(master_path)
    trained_state_dict = torch.load(model_filename, map_location="cpu")
    melu.load_state_dict(trained_state_dict)
    melu.cuda()

    eval_set_size = int(len(os.listdir("{}/item_cold_state".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in range(eval_set_size):
        supp_xs_s.append(pickle.load(open("{}/item_cold_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/item_cold_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/item_cold_state/query_x_{}.pkl".format(master_path, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/item_cold_state/query_y_{}.pkl".format(master_path, idx), "rb")))
    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

    evaluate(melu, total_dataset)
