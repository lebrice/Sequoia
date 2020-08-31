from main import (all_methods, all_settings, methods_dict, settings_dict)


def test_no_collisions_in_method_names():
    assert len(methods_dict) == len(all_methods)


def test_no_collisions_in_setting_names():
    assert len(settings_dict) == len(all_settings)
