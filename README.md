# <repo name>
Potential names:
- (ResearchTree?) (something tree-related?)


## Installation
Requires python >= 3.7

```console
git submodule init
git submodule update
pip install -r requirements.txt
```

## Getting Started:
- Take a look at Pytorch Lightning
- Take a quick look at [simple_parsing](https://github.com/lebrice/SimpleParsing) (A python package I've created) which we use to generate the command-line arguments for the experiments.
- Write a test that demonstrates how your new setting or method should behave.
- Implement your new method / setting
- [ ] Write some tests for **every single module**. Have them be easy to read so people could ideally understand how things work by simply reading the tests.
- Finally, graft your new method or setting onto the tree by adding them to `all_methods` or `all_settings`, respectively.

<!-- MAKETREE -->
   



## Registered Settings:

```
─ Setting (settings/base/setting.py)
├─── PassiveSetting (settings/passive/setting.py)
│  └─── ClassIncrementalSetting (settings/passive/cl/setting.py)
│     └─── TaskIncrementalSetting (settings/passive/cl/task_incremental/setting.py)
│        └─── IIDSetting (settings/passive/cl/task_incremental/iid/setting.py)
└─── ActiveSetting (settings/active/setting.py)
   └─── ContinualRLSetting (settings/active/rl/continual_rl_setting.py)
      └─── ClassIncrementalRLSetting (settings/active/rl/class_incremental_rl_setting.py)
         └─── TaskIncrementalRLSetting (settings/active/rl/task_incremental_rl_setting.py)
            └─── RLSetting (settings/active/rl/iid_rl_setting.py)
```


## Registered Methods:

## Methods: 

* [BaselineMethod](methods/baseline.py)
     Target setting: <class 'settings.base.setting.Setting'>
* [RandomBaselineMethod](methods/random_baseline.py)
     Target setting: <class 'settings.base.setting.Setting'>
* [SelfSupervision](methods/self_supervision.py)
     Target setting: <class 'settings.base.setting.Setting'>


## Registered Settings (with applicable methods): 

```
─ Setting (settings/base/setting.py)
│   Applicable methods: 
│    * [BaselineMethod](methods/baseline.py)
│    * [RandomBaselineMethod](methods/random_baseline.py)
│    * [SelfSupervision](methods/self_supervision.py)
│   
├─── PassiveSetting (settings/passive/setting.py)
│  │   Applicable methods: 
│  │    * [BaselineMethod](methods/baseline.py)
│  │    * [RandomBaselineMethod](methods/random_baseline.py)
│  │    * [SelfSupervision](methods/self_supervision.py)
│  │   
│  └─── ClassIncrementalSetting (settings/passive/cl/setting.py)
│     │   Applicable methods: 
│     │    * [BaselineMethod](methods/baseline.py)
│     │    * [RandomBaselineMethod](methods/random_baseline.py)
│     │    * [SelfSupervision](methods/self_supervision.py)
│     │   
│     └─── TaskIncrementalSetting (settings/passive/cl/task_incremental/setting.py)
│        │   Applicable methods: 
│        │    * [BaselineMethod](methods/baseline.py)
│        │    * [RandomBaselineMethod](methods/random_baseline.py)
│        │    * [SelfSupervision](methods/self_supervision.py)
│        │   
│        └─── IIDSetting (settings/passive/cl/task_incremental/iid/setting.py)
│               Applicable methods: 
│                * [BaselineMethod](methods/baseline.py)
│                * [RandomBaselineMethod](methods/random_baseline.py)
│                * [SelfSupervision](methods/self_supervision.py)
│               
└─── ActiveSetting (settings/active/setting.py)
   │   Applicable methods: 
   │    * [BaselineMethod](methods/baseline.py)
   │    * [RandomBaselineMethod](methods/random_baseline.py)
   │    * [SelfSupervision](methods/self_supervision.py)
   │   
   └─── ContinualRLSetting (settings/active/rl/continual_rl_setting.py)
      │   Applicable methods: 
      │    * [BaselineMethod](methods/baseline.py)
      │    * [RandomBaselineMethod](methods/random_baseline.py)
      │    * [SelfSupervision](methods/self_supervision.py)
      │   
      └─── ClassIncrementalRLSetting (settings/active/rl/class_incremental_rl_setting.py)
         │   Applicable methods: 
         │    * [BaselineMethod](methods/baseline.py)
         │    * [RandomBaselineMethod](methods/random_baseline.py)
         │    * [SelfSupervision](methods/self_supervision.py)
         │   
         └─── TaskIncrementalRLSetting (settings/active/rl/task_incremental_rl_setting.py)
            │   Applicable methods: 
            │    * [BaselineMethod](methods/baseline.py)
            │    * [RandomBaselineMethod](methods/random_baseline.py)
            │    * [SelfSupervision](methods/self_supervision.py)
            │   
            └─── RLSetting (settings/active/rl/iid_rl_setting.py)
                   Applicable methods: 
                    * [BaselineMethod](methods/baseline.py)
                    * [RandomBaselineMethod](methods/random_baseline.py)
                    * [SelfSupervision](methods/self_supervision.py)
                   
```
