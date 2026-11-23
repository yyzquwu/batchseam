import json
import re
from shlex import quote
import sys

from .scan import scan_tree


_ASSERTIONS = {"series": ("assert_elementwise", "Series"), "frame": ("assert_batch_invariant", "DataFrame")}


def _shell_arg(arg):
    if sys.platform != "win32":
        return quote(arg)
    if re.fullmatch(r"[\w./:-]+", arg):
        return arg
    return "'" + re.sub(r"['\u2018\u2019\u201a\u201b]", r"\g<0>\g<0>", arg) + "'"


def _guidance(finding):
    mode, kind, target = (finding[key] for key in ("mode", "kind", "target"))
    confirm = ("Verify that this is a Polars call before applying this advice. "
               if finding["confidence"] == "ambiguous" else "")
    if kind == "named" and target and mode in _ASSERTIONS:
        target = f"./{target}" if target.startswith("-") else target
        argv = ["batchseam", "check", target, "FIXTURE.parquet"]
        replacements = "FIXTURE.parquet with a representative fixture"
        if mode == "series":
            argv.extend(["--series", "COLUMN"])
            replacements += " and COLUMN with the input column"
        message = (f"From the scan root, replace {replacements} and run the check on trusted source. "
                   "Add --save-repro DIR to save a failure as a pytest regression.")
        return dict(guidance=confirm + message, argv=argv, command=" ".join(map(_shell_arg, argv)))

    callback = "callback"
    if kind == "lambda":
        message = "Extract the lambda into a named function and use that function at this call site and in the test. "
    elif kind == "qualified":
        callback = finding["callback"]
        message = "Import the same module in your test and use the qualified callback directly. "
    elif kind == "bound":
        method = finding["callback"].rpartition(".")[2]
        callback = f"instance.{method}" if method.isidentifier() else "instance.method"
        message = ("For a bound method, create an instance with the same configuration in your test. "
                   "If the attribute names a module function, use that function directly. ")
    else:
        message = "Identify the callable and its input and output types at this call site. "

    if mode == "multi":
        message += ("Build aligned inputs from a representative fixture. Compare the whole-input output with the combined "
                    "outputs after slicing every input at the same row offsets, and check the call's output contract.")
    elif mode in _ASSERTIONS:
        assertion, dtype = _ASSERTIONS[mode]
        message += (f"For a {dtype}-to-{dtype} callback, import {assertion} from batchseam and call "
                    f"{assertion}({callback}, {mode}) with a representative fixture.")
    else:
        message += (f"For a Series-to-Series callback, use assert_elementwise({callback}, series) from batchseam. "
                    f"For a DataFrame-to-DataFrame callback, use assert_batch_invariant({callback}, frame) from batchseam. "
                    "For other signatures, test the full-input and aligned sliced-input results in your project.")
    return dict(guidance=confirm + message, argv=None, command=None)


def _display(value):
    return "".join(char if char.isprintable() else ascii(char)[1:-1] for char in str(value))


def scan_report(path, *, as_json):
    report = scan_tree(path)
    findings, errors = report["findings"], report["errors"]
    report["findings"] = [{**finding, **_guidance(finding)} for finding in findings]
    if as_json:
        print(json.dumps(report, allow_nan=False))
    else:
        status = "incomplete" if errors else "complete"
        print(f"Scan {status} | {report['files_scanned']} Python files scanned | {len(findings)} findings")
        print("Scan root", _display(report["root"]))
        for finding in report["findings"]:
            print(_display("{path}:{line}  {flag}=True  [{confidence}]  "
                           "receiver {receiver}  {kind} callback {callback}".format(**finding)))
            print("  " + _display(finding["guidance"]))
            if finding["command"]:
                print("  " + _display(finding["command"]))
        for error in errors:
            print("  Scan error", _display("{path}: {message}".format(**error)))
    return 2 if errors else 0
