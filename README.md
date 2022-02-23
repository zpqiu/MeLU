# MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation

Official [MELU](https://github.com/hoyeoplee/MeLU) implementation + some changes


## Usage
### Requirements
- python 3.6+
- pytorch 1.1+
- tqdm 4.32+
- pandas 0.24+

### Preparing dataset
It needs about 266 MB of hard disk space.
```python
import os
from data_generation import generate
master_path= "./ml"
if not os.path.exists("{}/".format(master_path)):
    os.mkdir("{}/".format(master_path))
    generate(master_path)
```

### Training a model
Our model needs support and query sets. The support set is for local update, and the query set is for global update.
```python
import torch
import pickle
from MeLU import MeLU
from options import config
from model_training import training
melu = MeLU(config)
model_filename = "{}/models.pkl".format(master_path)
if not os.path.exists(model_filename):
    # Load training dataset.
    training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in range(training_set_size):
        supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))
    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
    training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
else:
    trained_state_dict = torch.load(model_filename)
    melu.load_state_dict(trained_state_dict)
```

### Evaluation
The evaluation part code refers to the repo [MELU_pytorch](https://github.com/waterhorse1/MELU_pytorch).

After training, we can run "`python model_evaluation.py`" to evaluate the trained model on the cold-item data set.


## Citation
The cite of original paper is
```
@inproceedings{lee2019melu,
  title={MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation},
  author={Lee, Hoyeop and Im, Jinbae and Jang, Seongwon and Cho, Hyunsouk and Chung, Sehee},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1073--1082},
  year={2019},
  organization={ACM}
}
```