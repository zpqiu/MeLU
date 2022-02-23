config = {
    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    # cuda setting
    'use_cuda': True,
    # model setting
    'inner': 1,
    'lr': 1e-3,
    'local_lr': 0.01,
    'batch_size': 32,
    'num_epoch': 30,
    # candidate selection
    'num_candidate': 20,
}

states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
