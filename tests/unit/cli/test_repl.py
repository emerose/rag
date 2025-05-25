import types

from prompt_toolkit.completion import PathCompleter, WordCompleter

from rag.cli.cli import _get_repl_completer


def test_get_repl_completer_returns_merged():
    commands = {"exit": lambda: None, "clear": lambda: None}
    completer = _get_repl_completer(commands)
    # _MergedCompleter is not exported, so compare attributes
    assert hasattr(completer, "completers")
    sub_types = {type(c) for c in completer.completers}
    assert WordCompleter in sub_types
    assert PathCompleter in sub_types
