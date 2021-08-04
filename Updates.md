# Sequoia Updates



## August 5th, 2021

Todos:
- [ ] Create an issue for PyTorch 1.9 compatibility
- [ ] Refactoring Replay method
    - [ ] Finish the tests
    - [ ] Make a PR
- [ ] Add PackNet to the PL example
    - [ ] Create a notebook version?
- [ ] Update the README.md:
    - [ ] Guide users more directly to the examples
    - [ ] Remove extra stuff
    - [ ] Add some images maybe?
    - [ ] Logo? 

Updates:

- [ ] Brax compatibility is almost here!
- [ ] PyTorch-Lightning folks are interested! (yay!)
- 


## July 29th, 2021:

Updates:
- Chat with PyTorch-Lightning Flash maintainers
- PyTorch-Lightning's `Callback` is an easy way to add a "plugin" to any algo!
    - Probably a good idea to retire this "auxiliary task" API.


Todos: 
- [ ] Push it to Arxiv
    - [ ] Finish empirical section
        - [ ] SL (ewc)
- [ ] Look into BRAX for massively parallel RL environments
- [ ] Make SB3 methods work w/ batched envs
- [ ] Refactor Replay (based on BaseMethod)


## July 22nd 2021:
Updates:
1. New way to add methods to Sequoia!
2. CN-DPM is now available as a Method!
3. Integration of Mostafa's submission in the examples!

New Issues:
- [ ] Look into using [Brax](https://www.github.com/google/brax) for RL
- [X] Add pytorch-lightning example


## Jul 7th

### Before Arxiv:

- [ ] Holes in CSL study
	- [x] launch Experience_replay in Cifar10
- [ ] Holes in CRL study
	- [x] No online performance anywhere (except Monsterkong). 
        - [X] Set `monitor_training_performance` to `True` by default in RL
		- [X] Relauch everything?
			- [ ] if so, start a new workspace and copy Monsterkong runs
			    - (Not sure this is needed)
            - [x] Replace '0' with None in Wandb so the average shows a good average of the online performance
	- [ ] Half-Chettah
		- [X] ~~baseline not launched (maybe bc it's not continuous?)~~ (BaseMethod doesn't support continuous action spaces yet)
		- [x] ~~no sucessful DQN anywhere~~ Missing DQN runs (DQN doesn't support continuous action spaces):
		    - [ ] CartPole
		    - [ ] MonsterKong
		- [ ] no sucessful SAC runs in multi-task and incremental RL:
		    - ![](https://i.imgur.com/YLxeGKW.png)
	- [x] MontainCar
		- ~~[ ] baseline method not launched (maybe bc it's not continuous?)~~ (same as above)
	- [ ] MonsterKong 
		- [ ] base method not launched
		- [ ] DQN
- [x] accept/reject updates in the overleaf
- [ ] Refactor Replay (based on BaseMethod)

### After Arxiv:
- [ ] Improved Command-Line API
- [ ] Debugging MetaWorld:
    - [ ] CW10 / MT10 / CW20 only have one run per algo?
        - Q: Are some properties persisting between runs in an hparam sweep? (e.g. train_env?)
    - [ ] SAC
        - [ ] IncrementalRL doesn't have runs
        - [ ] Step limit doesn't seem to be working
            - Q: is `max_episode_steps` being set on the Setting when using a MetaWorld end?
            - Do all metaworld envs have the same episode length limit? (500?)
- [ ] Add SAC Output Head to BaseMethod
- [ ] Choose the best name for 'Model' below:
    ```python3
    class BaseMethod:
        model: BaselineModel
    class Model(LightningModule):  # <--- this
        ...
    class MultiHeadModelMixin(Model):
        ...
    class SelfSupervisedMixin(Model):
        ...
    class SemiSupervisedMixin(Model):
        ...
    class BaseModel(
            MultiHeadModelMixin,
            SelfSupervisedModelMixin,
            SemiSupervisedModelMixin
        ):
        ...
        ```
- [x] Add 'avalanche' prefix to all avalanche methods, not just conflicting ones.
    - [x] Same, but for SB3 (or all Methods who have a 'family' field)
- [ ] Convert older runs in W&B:
    - [ ] Renamed settings
    - [ ] Renamed methods
- [ ] [reproducibility](https://github.com/lebrice/Sequoia/projects/12#card-64649672)

## June 7th:
Things to read:
- [ ] BabyAI: GridWorld + text
    - [ ] Paper https://openreview.net/pdf?id=rJeXCo0cYX
    - [ ] Code




# Massimo's Sequoia TODOs


### Before Arxiv

- finish CSL analysis
	- maybe add a multiple setting figure, e.g. w/ baseMethod
- finish CRL analysis
	- maybe add a multiple setting figure, e.g. w/ baseMethod