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

─ Setting (settings/base/setting.py)
├─── PassiveSetting (settings/passive/setting.py)
│  └─── ClassIncrementalSetting (settings/passive/cl/setting.py)
│     └─── TaskIncrementalSetting (settings/passive/cl/task_incremental/setting.py)
│        └─── IIDSetting (settings/passive/cl/task_incremental/iid/setting.py)
└─── ActiveSetting (settings/active/setting.py)
   └─── RLSetting (settings/active/rl/setting.py)


## Registered Methods:

## Methods: 

* [RandomBaselineMethod](methods/random_baseline.py)

     Target setting: <class 'settings.base.setting.Setting'>
* [ClassIncrementalMethod](methods/class_incremental_method.py)
     Target setting: <class 'settings.passive.cl.setting.ClassIncrementalSetting'>
* [TaskIncrementalMethod](methods/task_incremental_method.py)
     Target setting: <class 'settings.passive.cl.task_incremental.setting.TaskIncrementalSetting'>


## Registered Settings (with applicable methods): 
```
─ Setting (settings/base/setting.py)
│   Applicable methods: 
│    * [RandomBaselineMethod](methods/random_baseline.py)
│   
├─── PassiveSetting (settings/passive/setting.py)
│  │   Applicable methods: 
│  │    * [RandomBaselineMethod](methods/random_baseline.py)
│  │   
│  └─── ClassIncrementalSetting (settings/passive/cl/setting.py)
│     │   Applicable methods: 
│     │    * [RandomBaselineMethod](methods/random_baseline.py)
│     │    * [ClassIncrementalMethod](methods/class_incremental_method.py)
│     │   
│     └─── TaskIncrementalSetting (settings/passive/cl/task_incremental/setting.py)
│        │   Applicable methods: 
│        │    * [RandomBaselineMethod](methods/random_baseline.py)
│        │    * [ClassIncrementalMethod](methods/class_incremental_method.py)
│        │    * [TaskIncrementalMethod](methods/task_incremental_method.py)
│        │   
│        └─── IIDSetting (settings/passive/cl/task_incremental/iid/setting.py)
│               Applicable methods: 
│                * [RandomBaselineMethod](methods/random_baseline.py)
│                * [ClassIncrementalMethod](methods/class_incremental_method.py)
│                * [TaskIncrementalMethod](methods/task_incremental_method.py)
│               
└─── ActiveSetting (settings/active/setting.py)
   │   Applicable methods: 
   │    * [RandomBaselineMethod](methods/random_baseline.py)
   │   
   └─── RLSetting (settings/active/rl/setting.py)
          Applicable methods: 
           * [RandomBaselineMethod](methods/random_baseline.py)
          
```