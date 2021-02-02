# Sequoia - Methods

### Adding a new Method:

#### Prerequisites:
**- First, please take a look at the [examples](examples/)**

#### Steps:

1. Choose a target setting from the tree (See the "Available Settings" section below).

2. Create a new subclass of [`Method`](settings/base/bases.py), with the chosen target setting.

    Your class should implement the following methods:
    - `fit(train_env, valid_env)`
    - `get_actions(observations, action_space) -> Actions`
    
    The following methods are optional, but can be very useful to help customize how your method is used at train/test time:
    - `configure(setting: Setting)`
    - `on_task_switch(task_id: Optional[int])`
    - `test(test_env)`

    ```python
    class MyNewMethod(Method, target_setting=ClassIncrementalSetting):
        ... # Your code here.

        def fit(self, train_env: DataLoader, valid_env: DataLoader):
            # Train your model however you want here.
            self.trainer.fit(
                self.model,
                train_dataloader=train_env,
                val_dataloaders=valid_env,
            )
        
        def get_actions(self,
                        observations: Observations,
                        observation_space: gym.Space) -> Actions:
            # Return an "Action" (prediction) for the given observations.
            # Each Setting has its own Observations, Actions and Rewards types,
            # which are based on those of their parents.
            return self.model.predict(observations.x)

        def on_task_switch(self, task_id: Optional[int]):
            #This method gets called if task boundaries are known in the current
            #setting. Furthermore, if task labels are available, task_id will be
            # the index of the new task. If not, task_id will be None.
            # For example, you could do something like this:
            self.model.current_output_head = self.model.output_heads[task_id]
    ```

3. Running / Debugging your method:
 
    (at the bottom of your script, for example)

    ```python
    if __name__ == "__main__":
        ## 1. Create the setting you want to apply your method on.
        # First option: Create the Setting directly in code:
        setting = ClassIncrementalSetting(dataset="cifar10", nb_tasks=5)
        # Second option: Create the Setting from the command-line:
        setting = ClassIncrementalSetting.from_args()
        
        ## 2. Create your Method, however you want.
        my_method = MyNewMethod()

        ## 3. Apply your method on the setting to obtain results.
        results = setting.apply(my_method)
        # Optionally, display the results.
        print(results.summary())
        results.make_plots()
    ```

4. (WIP): Adding your new method to the tree:

    - Place the script/package that defines your Method inside of the `methods` folder.

    - Add the `@register_method` decorator to your Method definition, for example:

        ```python
        from sequoia.methods import register_method

        @register_method
        class MyNewMethod(Method, target_setting=ClassIncrementalSetting):
            name: ClassVar[str] = "my_new_method"
            ...
        ```

    - To launch an experiment using your method, run the following command:

        ```console
        python main.py --setting <some_setting_name> --method my_new_method
        ```
        To customize how your method gets created from the command-line, override the two following class methods:
        - `add_argparse_args(cls, parser: ArgumentParser)`
        - `from_argparse_args(cls, args: Namespace) -> Method`

    - Create a `<your_method_script_name>_test.py` file next to your method script. In it, write unit tests for every module/component used in your Method. Have them be easy to read so people can ideally understand how the components of your Method work by simply reading the tests.

        - (WIP) To run the unittests locally, use the following command: `pytest methods/my_new_method_test.py`

    - Then, write a functional test that demonstrates how your new method should behave, and what kind of results it expects to produce. The easiest way to do this is to implement a `validate_results(setting: Setting, results: Results)` method.
        - (WIP) To debug/run the "integration tests" locally, use the following command: `pytest -x methods/my_new_method_test.py --slow`

    - Create a Pull Request, and you're good to go!


## Registered Methods (so far):

- ## [BaselineMethod](baseline_method.py) 

	 - Target setting: [Setting](sequoia/settings/base/setting.py)

	Versatile Baseline method which targets all settings.

	Uses pytorch-lightning's Trainer for training and LightningModule as model. 

	Uses a [BaselineModel](methods/models/baseline_model/baseline_model.py), which
	can be used for:
	- Self-Supervised training with modular auxiliary tasks;
	- Semi-Supervised training on partially labeled batches;
	- Multi-Head prediction (e.g. in task-incremental scenario);

- ## [RandomBaselineMethod](random_baseline.py) 

	 - Target setting: [Setting](sequoia/settings/base/setting.py)

	Baseline method that gives random predictions for any given setting.

	This method doesn't have a model or any parameters. It just returns a random
	action for every observation.

- ## [EWC](ewc_method.py) 

	 - Target setting: [IncrementalSetting](sequoia/settings/assumptions/incremental.py)

	Minimal example of a Method targetting the Class-Incremental CL setting.

	For a quick intro to dataclasses, see examples/dataclasses_example.py    

- ## [DQNMethod](stable_baselines3_methods/dqn.py) 

	 - Target setting: [ContinualRLSetting](sequoia/settings/active/continual/continual_rl_setting.py)

	Method that uses a DQN model from the stable-baselines3 package. 

- ## [A2CMethod](stable_baselines3_methods/a2c.py) 

	 - Target setting: [ContinualRLSetting](sequoia/settings/active/continual/continual_rl_setting.py)

	Method that uses the DDPG model from stable-baselines3. 

- ## [DDPGMethod](stable_baselines3_methods/ddpg.py) 

	 - Target setting: [ContinualRLSetting](sequoia/settings/active/continual/continual_rl_setting.py)

	Method that uses the DDPG model from stable-baselines3. 

- ## [TD3Method](stable_baselines3_methods/td3.py) 

	 - Target setting: [ContinualRLSetting](sequoia/settings/active/continual/continual_rl_setting.py)

	Method that uses the TD3 model from stable-baselines3. 

- ## [SACMethod](stable_baselines3_methods/sac.py) 

	 - Target setting: [ContinualRLSetting](sequoia/settings/active/continual/continual_rl_setting.py)

	Method that uses the SAC model from stable-baselines3. 

- ## [PPOMethod](stable_baselines3_methods/ppo.py) 

	 - Target setting: [ContinualRLSetting](sequoia/settings/active/continual/continual_rl_setting.py)

	Method that uses the PPO model from stable-baselines3. 

- ## [ExperienceReplayMethod](experience_replay.py) 

	 - Target setting: [ClassIncrementalSetting](sequoia/settings/passive/cl/class_incremental_setting.py)

	Simple method that uses a replay buffer to reduce forgetting.


