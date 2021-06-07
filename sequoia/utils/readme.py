import os
import textwrap
from contextlib import nullcontext, redirect_stdout
from inspect import getsourcefile
from io import StringIO
from pathlib import Path
from typing import List, Type

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sequoia.methods import Method
    from sequoia.settings import Setting


def get_relative_path_to(something: Type) -> Path:
    """ Attempts to give the relative path from the current working directory to the
    file where somethign is defined. If that's not possible, returns an absolute path
    instead.
    """
    source_dir = Path.cwd().parent
    source_file = Path(getsourcefile(something)).relative_to(source_dir)
    return source_file


def get_tree_string(
    root_setting: Type["Setting"] = "Setting",
    with_methods: bool = False,
    with_assumptions: bool = False,
    with_docstrings: bool = False,
) -> str:
    """Get a string representation of the tree!
    
    I want to return something like this:
    ```
    "Setting"
    ├── active
    │   └── rl
    ├── base
    └── passive
        └── cl
            └── task_incremental
                └── iid
    ```
    """
    if with_assumptions:
        raise NotImplementedError(
            f"TODO: display the assumptions for each setting into the tree string "
            f"somehow."
        )
    setting: Type["Setting"] = root_setting
    prefix: str = ""

    message: List[str] = []
    source_file = get_relative_path_to(setting)
    message += [f"{setting.__name__} ({source_file})"]

    applicable_methods = setting.get_applicable_methods()

    n_children = len(setting.get_immediate_children())
    bar = "│" if n_children else " "

    if with_docstrings:
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
    # print(f"Children: {setting.get_children()}")
    # print(f"Children[0]'s children: {setting.get_children()[0].children}")

    for i, child_setting in enumerate(setting.get_immediate_children()):
        # Recurse!
        child_message = get_tree_string(child_setting)

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


def get_tree_string_markdown(
    root_setting: Type["Setting"] = "Setting",
    with_methods: bool = False,
    with_docstring: bool = False,
):
    """Get a string representation of the tree!
    
    I want to return something like this:
    
    - "Setting"
        - active
            - rl
    - base
        - passive
            - cl
                - task_incremental
                    * iid
    
    """
    setting = root_setting

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
    # print(f"Children: {setting.get_children()}")
    # print(f"Children[0]'s children: {setting.get_children()[0].children}")

    for child_setting in setting.get_immediate_children():
        child_message = get_tree_string_markdown(
            child_setting, with_methods=with_methods, with_docstring=with_docstring
        )
        child_message = textwrap.indent(child_message, tab)
        message_lines += [""]
        message_lines.extend(child_message.splitlines())
        message_lines += [""]

    return "\n".join(message_lines)


def print_methods():
    for method in all_methods:
        source_file = get_relative_path_to(method)
        target_setting: Type["Setting"] = method.target_setting
        setting_file = get_relative_path_to(target_setting)
        print(f"- ## [{method.__name__}]({source_file}) ")
        print()
        print(f"\t - Target setting: [{target_setting.__name__}]({setting_file})")
        print()
        docstring: str = method.__doc__
        docstring_lines = docstring.splitlines()
        # The first line is always less indented than the rest, which looks weird:
        first_line = docstring_lines[0].lstrip()
        # Remove the common indent in the rest of the docstring lines:
        other_lines = textwrap.dedent("\n".join(docstring_lines[1:]))
        # re-indent the docstring, with all equal indentation now:
        docstring = first_line + "\n" + other_lines
        print(textwrap.indent(docstring, "\t"))


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
            print(*lines[: tree_index + 1], sep="")
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
    print(get_tree_string())
    # add_stuff_to_readme()
