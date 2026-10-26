"""Small tools for changing config.toml as TEXT, so the comments and layout are kept.

Used by the results app (src/app/build/) and ospfmusim/run.py.

Only one-line values (numbers, booleans, strings, one-line lists) in [section]
tables can be changed. Multi-line arrays (such as mass_matrix) and [[array
tables]] (such as [[thrusters]]) are left as they are.
"""
import json
import re
import tomllib
from pathlib import Path

SECTION = re.compile(r"^\s*\[(?P<name>[^\[\]]+)\]\s*(#.*)?$")
ARRAY_TABLE = re.compile(r"^\s*\[\[")
KEY_LINE = re.compile(r"^(?P<prefix>\s*(?P<key>[A-Za-z0-9_]+)\s*=\s*)(?P<rest>.*)$")


def split_comment(rest):
    """Splits 'value   # comment' into ('value', '   ', '# comment'). A # inside quotes is kept."""
    quote = None
    for i, char in enumerate(rest):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            value = rest[:i].rstrip()
            return value, rest[len(value):i], rest[i:]
    return rest.rstrip(), "", ""


def lines_with_keys(text):
    """Yields (line, section, key, prefix, value, spacing, comment) for every line.

    section, key, ... are None for lines that are not editable one-line values.
    """
    section = None
    for line in text.splitlines():
        if ARRAY_TABLE.match(line):
            section = None                       # [[...]] tables are not editable
        elif match := SECTION.match(line):
            section = match["name"].strip()
        elif (section is not None and not line.lstrip().startswith("#")
              and (match := KEY_LINE.match(line))):
            value, spacing, comment = split_comment(match["rest"])
            if value.startswith("[") and not value.endswith("]"):
                yield line, None, None, None, None, None, None      # start of a multi-line array
                continue
            yield line, section, match["key"], match["prefix"], value, spacing, comment
            continue
        yield line, None, None, None, None, None, None


def format_value(value):
    """A Python value as TOML text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 1e6 or abs(value) < 1e-3):
            # Shortest exact scientific notation: 1e7 -> "1e7", 7.6e-5 -> "7.6e-5".
            for digits in range(17):
                mantissa, exponent = f"{value:.{digits}e}".split("e")
                if float(f"{mantissa}e{exponent}") == value:
                    return f"{mantissa}e{int(exponent)}"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)                 # a TOML basic string for plain text
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} to TOML")


def set_values(text, changes):
    """config text with new values: changes = {("section", "key"): value}.

    The comment on each changed line is kept, at the same column if there is room.
    Raises KeyError if a key is not found as a one-line value.
    """
    out, done = [], set()
    for line, section, key, prefix, value, spacing, comment in lines_with_keys(text):
        if (section, key) in changes:
            new = prefix + format_value(changes[(section, key)])
            if comment:
                column = len(prefix) + len(value) + len(spacing)
                new = new + " " * max(column - len(new), 1) + comment
            line = new
            done.add((section, key))
        out.append(line)
    missing = set(changes) - done
    if missing:
        raise KeyError(f"not found as one-line values in the config: {sorted(missing)}")
    result = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    tomllib.loads(result)                        # fails here if the result is not valid TOML
    return result


def comments(text):
    """{("section", "key"): inline comment without '#'}, used as help texts."""
    return {(section, key): comment.lstrip("#").strip()
            for _, section, key, _, _, _, comment in lines_with_keys(text) if key and comment}


def make_paths_absolute(text, base_dir):
    """Turns relative file paths (keys ending in '_file') into absolute ones, relative to base_dir.

    Needed when a copy of the config is saved in another folder, because the
    FMUs read those paths relative to the config file.
    """
    changes = {}
    for _, section, key, _, value, _, _ in lines_with_keys(text):
        if key and key.endswith("_file"):
            path = tomllib.loads(f"v = {value}")["v"]
            if not Path(path).is_absolute():
                changes[(section, key)] = (Path(base_dir) / path).resolve().as_posix()
    return set_values(text, changes) if changes else text
