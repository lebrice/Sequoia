"""TODO (@lebrice): Create an example of a new Method class.
"""
from methods import Method, register_method
from examples.new_setting import NewSetting


@register_method
class NewMethod(Method, target_setting=NewSetting):
    """ Example of a new Method aimed at solving the new research setting. """

    def apply_to(self, setting):
        # 1. Configure the method to work on the setting.
        self.configure(setting)
        # 2. Train the method on the setting.
        self.train(setting)
        # 3. Evaluate the model on the setting and return the results.
        return setting.evaluate(self)

if __name__ == "__main__":
    NewMethod.main()
