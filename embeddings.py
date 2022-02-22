import torch
import torch.nn as nn
import torch.nn.functional as F


class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.num_director = config['num_director']
        self.num_actor = config['num_actor']
        self.embedding_dim = config['embedding_dim']

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate + 1, 
            embedding_dim=self.embedding_dim
        )

        self.embedding_genre = torch.nn.Embedding(
            num_embeddings=self.num_genre + 1, 
            embedding_dim=self.embedding_dim
        )

        self.embedding_director = torch.nn.Embedding(
            num_embeddings=self.num_director + 1, 
            embedding_dim=self.embedding_dim
        )

        self.embedding_actor = torch.nn.Embedding(
            num_embeddings=self.num_actor + 1, 
            embedding_dim=self.embedding_dim
        )

    def multi_valued_feature_pooling(self, feature, feature_embedding):
        # feature: [*, feature_max_len]
        mask = torch.gt(feature, 0).float()
        value_counts = torch.sum(mask, dim=-1, keepdim=True) + 1e-8
        mean_pooling = torch.sum(feature_embedding * mask.unsqueeze(-1), dim=1) / value_counts
        return mean_pooling

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.multi_valued_feature_pooling(genre_idx, self.embedding_genre(genre_idx))
        director_emb = self.multi_valued_feature_pooling(director_idx, self.embedding_director(director_idx))
        actors_emb = self.multi_valued_feature_pooling(actors_idx, self.embedding_actor(actors_idx))
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']
        self.embedding_dim = config['embedding_dim']

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender + 1,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age + 1,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation + 1,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode + 1,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
