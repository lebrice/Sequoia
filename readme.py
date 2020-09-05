
import os
import textwrap
from contextlib import redirect_stdout
from inspect import getsourcefile
from pathlib import Path
from typing import List, Type

from methods import Method, all_methods
from settings import Setting, all_settings


def get_relative_path_to(something: Type):
    cwd = Path(os.path.abspath(os.path.dirname(__file__)))
    source_file = Path(getsourcefile(something)).relative_to(cwd)
    return source_file

def get_tree_string(with_methods: bool = False):
    """Get a string representation of the tree!
    
    I want to return something like this:
    ```
    Setting
    ├── active
    │   └── rl
    ├── base
    └── passive
        └── cl
            └── task_incremental
                └── iid
    ```
    """
    def _setting_tree(setting: Type[Setting], prefix: str = "", indentation: int=0) -> str:
        message: List[str] = []
        source_file = get_relative_path_to(setting)
        message += [f"{setting.__name__} ({source_file})"]

        applicable_methods = setting.get_all_applicable_methods()

        n_children = len(setting.sub_settings)
        bar = "│" if n_children else " "
        
        if with_methods:
            p = f"{bar}  "
            message += [f"{p} Applicable methods: "]
            for method in applicable_methods:
                source_file = get_relative_path_to(method)
                message += [f"{p}  * [{method.__name__}]({source_file})"]
            message += [f"{p} "]
        
        # message = "\n".join(message) + "\n"
        # print(f"Children: {setting._sub_settings}")
        # print(f"Children[0]'s children: {setting._sub_settings[0]._sub_settings}")
        
        for i, child_setting in enumerate(setting.sub_settings):
            child_prefix = prefix + ""
            # Recurse!
            child_message = _setting_tree(child_setting, child_prefix, indentation + 1)
            
            child_message_lines = child_message.splitlines()
            for j, line in enumerate(child_message_lines):
                first: str = "x  "
                if j == 0:
                    if i == n_children - 1:
                        first = "└──"
                    else:
                        first = "├──"
                else:
                    if i == n_children - 1:
                        first = "   "
                    else:
                        first = "│  "
                message += [first + prefix + line]

        first_line = f"─ {message[0]}\n"
        message = "\n".join(message[1:])
        message = textwrap.indent(message, prefix)
        return first_line + message
    return _setting_tree(Setting)

def print_methods():
    print("## Methods: \n")
    for method in all_methods:
        source_file = get_relative_path_to(method)
        print(f"* [{method.__name__}]({source_file})")
        print(f"     Target setting: {method.target_setting}")

def add_stuff_to_readme(readme_path=Path("README.md")):
    token = "<!-- MAKETREE -->"

    lines: List[str] = []
    from io import StringIO
    with open(readme_path) as f:
        with StringIO(f.read()) as f:
            lines = f.readlines()
            if token not in lines:
                print("didn't find token!")
                exit()
            tree_index = lines.index(token) + 1

    with open(readme_path, "w") as f:
        with redirect_stdout(f):
            # reversed insert?
            print(*lines[:tree_index + 1])
            print("\n\n## Registered Settings:\n")
            print(get_tree_string())
            print("\n\n## Registered Methods:\n")
            print_methods()
            print("\n\n## Registered Settings (with applicable methods): \n")
            print(get_tree_string(with_methods=True))

if __name__ == "__main__":
    add_stuff_to_readme()
