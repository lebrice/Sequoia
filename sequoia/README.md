# sequoia

## Packages:
- [settings](sequoia/settings): definitions for the settings (machine learning problems).
- [methods](sequoia/methods): Contains the methods (which can be applied to settings).
- [common](sequoia/common): utilities such as metrics, transforms, layers, gym wrappers configuration classes, etc. that are used by Settings and Methods.
- [utils](sequoia/common): miscelaneous utility functions (logging, command-line parsing, etc)
- [experiments](sequoia/experiments): Command-line interface entry-points, via the `Experiment` class.
- [client (wip)](sequoia/client): defines a proxy to a Setting and its environments, in order to further isolate the Method and Setting from each other (used for the CLVision competition). 
