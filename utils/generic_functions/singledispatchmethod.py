""" Little 'patch' that imports a backport of 'singledispatchmethod', if the
python version is < 3.8.
"""
import sys
if sys.version_info >= (3, 8):
    from functools import singledispatchmethod  # type: ignore
else:
    try:
        from singledispatchmethod import singledispatchmethod
    except ImportError as e:
        print(f"Couldn't import singledispatchmethod: {e}")
        print("Since you're running python version below 3.8, you need to "
              "install the backport for singledispatchmethod (which was added "
              "to functools in python 3.8), using the following command:\n"
              "> pip install singledispatchmethod")
        exit()