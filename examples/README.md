# Examples

Here's a brief description of the examples in this folder:

## Prerequisites:
- [Intro to dataclasses & simple-parsing](prerequisites/dataclasses_example.py)
- [Basics of openai gym](https://github.com/openai/gym#basics)


## Basic examples:

- [quick_demo.ipynb](basic/quick_demo.ipynb):
    **Recommended entry-point for new users**. Simple demo showing how to create a `Method`
    from scratch that targets a Supervised CL `Setting`, as well as how to
    improve this simple Method using a simple regularization loss.

    - [quick_demo.py](basic/quick_demo.py): First part of the above
        notebook: shows how to create a Method from scratch that
        targets a Supervised CL Setting.
    - [quick_demo_ewc.py](basic/quick_demo_ewc.py): Second part of the
        above notebook: shows how to improve upon an existing Method by adding a
        CL regularization loss.

- [baseline_demo.py](basic/baseline_demo.py): Shows how the
    BaselineMethod can be applied to get results in both RL and SL Settings.


## CLVision Workshop Submission Examples:

- [Simple Classifier](clcomp21/classifier.py):
    Standard neural net classifier without any CL-related mechanism. Works in the SL track, but has very poor performance.

- [Multi-Head / Task Inference Classifier](clcomp21/multihead_classifier.py):
    Performs multi-head prediction, and a simple form of task inference. Gets better results that the example.

- (wip) [CL Regularized Classifier](clcomp21/regularization_example.py):
    Adds a simple CL regularization loss to the multihead classifier above.

- (More to be added shortly)


## Advanced examples:

- [RL_and_SL_demo.py](advanced/RL_and_SL_demo.py):
    
    Example that shows how the BaselineMethod can easily be extended by adding
    AuxiliaryTasks to it, allows you to get results in both RL and SL.

- [continual_rl_demo.py](advanced/ewc_in_rl.py):
    
    Demonstrates how to create Reinforcement Learning (RL) Settings, as well as
    how methods from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
    can be applied to these settings.


- [Extending Stable-Baselines3 (RL Settings only)](advanced/ewc_in_rl.py):

    (Not recommended for new users!)
    Very specific example which shows how, if you really wanted to, you could
    extend one or more of the Methods from SB3 with some kind of regularization
    loss hooking into the internal optimization loop of SB3.
