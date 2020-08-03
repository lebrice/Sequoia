
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from common.losses import LossInfo, get_supervised_accuracy
import numpy as np
from typing import Tuple
import wandb
from utils.logging_utils import log_wandb

def background_eval(q, config): 
    wandb.init( 
            project=config.project_name,
            name=config.run_name+'_linear',
            group=config.run_group,
            #dir=str(config.wandb_path),
            reinit=True,
            tags=config.tags,
            #resume="allow",
            job_type='background'
        )
    while True:
        item =q.get()
        if item is None:
            break
        get_mlp_lossinfo(*item)

def get_mlp_lossinfo(i, X, y, Xt, yt, step, description=''):
    linear_i_train_loss, linear_i_test_loss = get_MLP_losses(X, y, Xt, yt, description='')
    print("done training linear")
    linear_i_train_acc = get_supervised_accuracy(linear_i_train_loss)
    linear_i_test_acc = get_supervised_accuracy(linear_i_test_loss)
    wandb.log({
        f"{description}/train/task{i}": linear_i_train_acc,
        f"{description}/test/task{i}" : linear_i_test_acc,
        f"{description}/test": linear_i_test_acc,
    })

def get_MLP_losses(X, y, Xt, yt, random_state=100, description='') -> Tuple[LossInfo, LossInfo]:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xt = scaler.transform(Xt)
    clf = MLPClassifier(random_state=random_state, max_iter=100).fit(X, y)
    classes = clf.classes_
    y_prob = clf.predict_proba(X)
    y_t_prob = clf.predict_proba(Xt)

    y_logits = np.zeros((y.size, y.max() + 1))
    for i, label in enumerate(classes):
        y_logits[:, label] = y_prob[:, i]

    y_t_logits = np.zeros((yt.size, yt.max() + 1))
    for i, label in enumerate(classes):
        y_t_logits[:, label] = y_t_prob[:, i]
    

    nce = log_loss(y_true=y, y_pred=y_prob, labels=classes)
    nce_t = log_loss(y_true=yt, y_pred=y_t_prob, labels=classes)

    from tasks.tasks import Tasks
    train_loss = LossInfo(f"{description}_train")
    train_loss= train_loss + LossInfo(Tasks.SUPERVISED, total_loss=nce, y_pred=y_logits, y=y)

    test_loss = LossInfo(f"{description}_test")
    test_loss = test_loss + LossInfo(Tasks.SUPERVISED, total_loss=nce_t, y_pred=y_t_logits, y=yt)
    del X, Xt, yt, y

    return train_loss, test_loss