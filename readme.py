
import os
import textwrap
from contextlib import nullcontext, redirect_stdout
from inspect import getsourcefile
from io import StringIO
from pathlib import Path
from typing import List, Type

from methods import Method, all_methods
from settings import Setting, all_settings


def get_relative_path_to(something: Type):
    cwd = Path(os.path.abspath(os.path.dirname(__file__)))
    source_file = Path(getsourcefile(something)).relative_to(cwd)
    return source_file

def get_tree_string(with_methods: bool = False, with_docstring: bool = False):
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

        applicable_methods = setting.get_applicable_methods()

        n_children = len(setting.children)
        bar = "│" if n_children else " "
        
        if with_docstring:
            p = f"{bar}  "
            docstring = setting.__doc__
            message.extend([p + line for line in docstring.splitlines()])
            message += [p]

        if with_methods:
            p = f"{bar}  "
            message += [f"{p} Applicable methods: "]
            for method in applicable_methods:
                source_file = get_relative_path_to(method)
                message += [f"{p}  * [{method.__name__}]({source_file})"]
            message += [f"{p} "]
        
        # message = "\n".join(message) + "\n"
        # print(f"Children: {setting.children}")
        # print(f"Children[0]'s children: {setting.children[0].children}")
        
        for i, child_setting in enumerate(setting.children):
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



def get_tree_string_markdown(with_methods: bool = False, with_docstring: bool = False):
    """Get a string representation of the tree!
    
    I want to return something like this:
    
    - Setting
        - active
            - rl
    - base
        - passive
            - cl
                - task_incremental
                    * iid
    
    """

    def _setting_tree(setting: Type[Setting]) -> str:
        message_lines: List[str] = []
        source_file = get_relative_path_to(setting)
        message_lines += [f"- ## [{setting.__name__}]({source_file})"]

        applicable_methods = setting.get_applicable_methods()
        tab = "\t"

        if with_docstring:
            message_lines += [""]
            docstring: str = setting.__doc__
            docstring_lines = docstring.splitlines()
            # The first line is always less indented than the rest, which looks weird:
            first_line = docstring_lines[0].lstrip()
            # Remove the common indent in the rest of the docstring lines:
            other_lines = textwrap.dedent("\n".join(docstring_lines[1:]))
            # re-indent the docstring, with all equal indentation now:
            docstring = first_line + "\n" + other_lines
            # docstring = textwrap.shorten(docstring, replace_whitespace=False, width=130)
            # docstring = textwrap.fill(docstring, max_lines=10)
            # print(setting)
            # print(docstring)
            # exit()
            docstring = textwrap.indent(docstring, tab)

            message_lines.extend(docstring.splitlines())
            message_lines += [""]

        if with_methods:
            message_lines += [""]
            message_lines += ["Applicable methods: "]
            for method in applicable_methods:
                source_file = get_relative_path_to(method)
                message_lines += [f" * [{method.__name__}]({source_file})"]
            message_lines += [""]
        
        # message = "\n".join(message) + "\n"
        # print(f"Children: {setting.children}")
        # print(f"Children[0]'s children: {setting.children[0].children}")
        
        for child_setting in setting.children:
            child_message = _setting_tree(child_setting)
            child_message = textwrap.indent(child_message, tab)
            message_lines += [""]
            message_lines.extend(child_message.splitlines())
            message_lines += [""]

        return "\n".join(message_lines)
    return _setting_tree(Setting)




def print_methods():
    for method in all_methods:
        source_file = get_relative_path_to(method)
        print(f"* [{method.__name__}]({source_file})")
        target_setting: Type[Setting] = method.target_setting
        setting_file = get_relative_path_to(target_setting)
        print(f"     Target setting: [{target_setting.__name__}]({setting_file})")


def add_stuff_to_readme(readme_path=Path("README.md")):
    token = "<!-- MAKETREE -->\n"

    lines: List[str] = []
    with open(readme_path) as f:
        with StringIO(f.read()) as f:
            lines = f.readlines()
            if token not in lines:
                print("didn't find token!")
                exit()
            tree_index = lines.index(token) + 1
    
    # print(get_tree_string_markdown(with_methods=False, with_docstring=True))
    # exit()

    with open(readme_path, "w") as f:
    # with nullcontext():
        with redirect_stdout(f):
        # with nullcontext():
            # reversed insert?
            # Print the existing lines back:
            print(*lines[:tree_index + 1], sep="")
            print("\n\n## Available Settings:\n")
            print()
            print(get_tree_string_markdown(with_methods=False, with_docstring=True))
            print()
            # print("```")
            # print(get_tree_string())
            # print("```")
            print("\n\n## Registered Methods (so far):\n")
            print_methods()
            print()

if __name__ == "__main__":
    add_stuff_to_readme()
