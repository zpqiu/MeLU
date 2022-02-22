import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm

from options import states
from dataset import movielens_1m


def item_converting(row, rate_dict, genre_dict, director_dict, actor_dict):
    def _cut_and_padding(value_list, max_len=10):
        cutted_list = value_list[:max_len]
        return cutted_list + [0,] * (max_len - len(cutted_list))
    # Multi-valued
    rate_idx = torch.tensor([[rate_dict.get(str(row['rate']), 0)]]).long()
    genre_idx = list()
    for genre in str(row['genre']).split(", "):
        genre_idx.append(genre_dict.get(genre, 0))
    genre_idx = torch.tensor([_cut_and_padding(genre_idx), ]).long()

    director_idx = list()
    for director in str(row['director']).split(", "):
        director = re.sub(r'\([^()]*\)', '', director)
        director_idx.append(director_dict.get(director, 0))
    director_idx = torch.tensor([_cut_and_padding(director_idx), ]).long()

    actor_idx = list()
    for actor in str(row['actors']).split(", "):
        actor_idx.append(actor_dict.get(actor, 0))
    actor_idx = torch.tensor([_cut_and_padding(actor_idx), ]).long()

    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1) # [1, 31]


def user_converting(row, gender_dict, age_dict, occupation_dict, zipcode_dict):
    # Single-valued
    gender_idx = torch.tensor([[gender_dict.get(str(row['gender']), 0)]]).long()
    age_idx = torch.tensor([[age_dict.get(str(row['age']), 0)]]).long()
    occupation_idx = torch.tensor([[occupation_dict.get(str(row['occupation_code']), 0)]]).long()
    zip_idx = torch.tensor([[zipcode_dict.get(str(row['zip'])[:5], 0)]]).long()

    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1) # [1, 4]


def load_dict(fname):
    dict_ = dict()
    # 所有feature 的index = 0为padding
    with open(fname, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            dict_[line.strip()] = idx + 1
    return dict_


def generate(master_path):
    dataset_path = "movielens/ml-1m"
    rate_dict = load_dict("{}/m_rate.txt".format(dataset_path))
    genre_dict = load_dict("{}/m_genre.txt".format(dataset_path))
    actor_dict = load_dict("{}/m_actor.txt".format(dataset_path))
    director_dict = load_dict("{}/m_director.txt".format(dataset_path))
    gender_dict = load_dict("{}/m_gender.txt".format(dataset_path))
    age_dict = load_dict("{}/m_age.txt".format(dataset_path))
    occupation_dict = load_dict("{}/m_occupation.txt".format(dataset_path))
    zipcode_dict = load_dict("{}/m_zipcode.txt".format(dataset_path))

    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    dataset = movielens_1m()

    # hashmap for item information
    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, rate_dict, genre_dict, director_dict, actor_dict)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
    # hashmap for user profile
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, gender_dict, age_dict, occupation_dict, zipcode_dict)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_movie_len = len(dataset[str(u_id)])
            indices = list(range(seen_movie_len))

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue

            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            tmp_y = np.array(dataset_y[str(u_id)])

            support_x_app = None
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted

            query_x_app = None
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted
            support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
            query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1
