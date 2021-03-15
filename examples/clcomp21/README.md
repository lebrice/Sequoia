## Example Submissions for CLVision Workshop

Examples in this folder are aimed at solving the supervised learning track of the competition.

Each example builds on top of the previous, in a manner that improves the overall performance you can expect on any given CL setting.

As such, it is recommended that you take a look at the examples in the following order:

0. [DummyMethod](dummy_method.py)
    Non-parametric method that simply returns a random prediction for each observation.

1. [Simple Classifier](classifier.py):
    Standard neural net classifier without any CL-related mechanism. Works in the SL track, but has very poor performance.

2. [Multi-Head / Task Inference Classifier](multihead_classifier.py):
    Performs multi-head prediction, and a simple form of task inference. Gets better results that the example.

3. [CL Regularized Classifier](regularization_example.py):
    Adds a simple CL regularization loss to the multihead classifier above.
