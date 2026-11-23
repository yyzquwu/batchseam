"""Find explicit Polars batch promises without importing the inspected module."""

import ast
from dataclasses import dataclass


_EXPR = {"Expr", "all", "col", "element", "first", "last", "lit", "nth", "struct"}
_LAZY = {"LazyFrame", "scan_csv", "scan_delta", "scan_ipc", "scan_ndjson", "scan_parquet"}
_LAZY_METHODS = {
    "cache", "cast", "clone", "drop", "filter", "head", "join", "limit", "map_batches",
    "rename", "select", "slice", "sort", "tail", "unique", "with_columns",
}
_EXPR_METHODS = {"alias", "cast", "fill_nan", "fill_null", "map_batches", "over"}
_POLARS_MEMBERS = {
    "polars": _EXPR | _LAZY | {"DataFrame", "map_batches"},
    "polars.expr": {"Expr"}, "polars.expr.expr": {"Expr"},
    "polars.functions": (_EXPR - {"Expr"}) | {"map_batches"},
    "polars.functions.lazy": {"map_batches"},
    "polars.lazyframe": {"LazyFrame"}, "polars.lazyframe.frame": {"LazyFrame"},
    "polars.dataframe": {"DataFrame"}, "polars.dataframe.frame": {"DataFrame"},
}
_FLAGS = {"series": "is_elementwise", "multi": "is_elementwise", "frame": "streamable"}


@dataclass
class _Binding:
    node: ast.AST | None
    value: ast.AST | str | None
    annotation: ast.expr | None
    annotation_scope: "_Scope"


class _Scope:
    def __init__(self, parent=None, *, kind="module"):
        self.parent = parent
        self.kind = kind
        self.bindings = {}
        self.globals = set()
        self.has_yield = False


def _pos(node):
    return node.lineno, node.col_offset


def _positional(args):
    return args.posonlyargs + args.args


class _Scan(ast.NodeVisitor):
    def __init__(self):
        self.module = self.scope = _Scope()
        self.calls = []
        self.generators = set()
        self.conditional = 0

    def _bind(self, name, node, value=None, annotation=None, annotation_scope=None):
        self.scope.bindings.setdefault(name, []).append(_Binding(
            node, value if not self.conditional else None,
            annotation if not self.conditional else None, annotation_scope or self.scope,
        ))
        if name in self.scope.globals:
            self.module.bindings.setdefault(name, []).append(_Binding(node, None, None, self.module))

    def visit_Import(self, node):
        for item in node.names:
            name = item.asname or item.name.split(".")[0]
            module = item.name if item.asname else name
            self._bind(name, node, module if module in _POLARS_MEMBERS else None)

    def visit_ImportFrom(self, node):
        for item in node.names:
            name = f"{node.module}.{item.name}"
            value = name if not node.level and node.module in _POLARS_MEMBERS else None
            self._bind(item.asname or item.name, node, value)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id, None if self.scope.kind == "comprehension" else node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind(target.id, node, node.value)
            else:
                self.visit(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._bind(node.target.id, node, node.value, node.annotation)
            self.visit(node.annotation)
            self.visit(node.value)
        else:
            self.generic_visit(node)

    def visit_NamedExpr(self, node):
        scope = self.scope
        while self.scope.kind == "comprehension":
            self.scope = self.scope.parent
        self._bind(node.target.id, node)
        self.scope = scope
        self.visit(node.value)

    def visit_ExceptHandler(self, node):
        if node.name:
            self._bind(node.name, node)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self._bind(node.name, node)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest:
            self._bind(node.rest, node)
        self.generic_visit(node)

    def visit_Global(self, node):
        if self.scope is self.module:
            return
        for name in node.names:
            self.scope.bindings.setdefault(name, [])
        self.scope.globals.update(node.names)

    def visit_Nonlocal(self, node):
        for name in node.names:
            self._bind(name, node)

    def _branch(self, node):
        self.conditional += 1
        self.generic_visit(node)
        self.conditional -= 1

    visit_If = visit_For = visit_AsyncFor = visit_While = visit_With = visit_AsyncWith = _branch
    visit_Try = visit_TryStar = visit_Match = visit_IfExp = visit_BoolOp = _branch

    def _enter(self, kind):
        outer, conditional = self.scope, self.conditional
        parent = outer.parent if outer.kind == "class" else outer
        self.scope, self.conditional = _Scope(parent, kind=kind), 0
        return outer, conditional

    def _function(self, node):
        self.visit(node.args)
        outer, conditional = self._enter("function")
        args = node.args
        for arg in (*_positional(args), *args.kwonlyargs):
            self._bind(arg.arg, None, annotation=arg.annotation, annotation_scope=outer)
        for arg in (args.vararg, args.kwarg):
            if arg is not None:
                self._bind(arg.arg, None)
        if isinstance(node, ast.Lambda):
            self.visit(node.body)
        else:
            for stmt in node.body:
                self.visit(stmt)
        if self.scope.has_yield:
            self.generators.add(node)
        self.scope, self.conditional = outer, conditional

    def visit_FunctionDef(self, node):
        self._bind(node.name, node, node)
        for expr in (*node.decorator_list, *getattr(node, "type_params", ()), node.returns):
            if expr is not None:
                self.visit(expr)
        self._function(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = _function

    def visit_Yield(self, node):
        self.scope.has_yield = True
        self.generic_visit(node)

    visit_YieldFrom = visit_Yield

    def visit_ClassDef(self, node):
        self._bind(node.name, node)
        for expr in (*node.decorator_list, *node.bases, *node.keywords, *getattr(node, "type_params", ())):
            self.visit(expr)
        outer, conditional = self._enter("class")
        for stmt in node.body:
            self.visit(stmt)
        self.scope, self.conditional = outer, conditional

    def _comprehension(self, node):
        first, *rest = node.generators
        self.visit(first.iter)
        outer, conditional = self._enter("comprehension")
        for generator in (first, *rest):
            if generator is not first:
                self.visit(generator.iter)
            self.visit(generator.target)
            for expr in generator.ifs:
                self.visit(expr)
        for expr in (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,):
            self.visit(expr)
        self.scope, self.conditional = outer, conditional

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comprehension

    def visit_Call(self, node):
        flags = [kw.arg for kw in node.keywords if kw.arg in _FLAGS.values() and _literal(kw.value, True)]
        if flags:
            self.calls.append((node, self.scope, flags))
        self.generic_visit(node)


def _binding(name, scope, pos):
    while scope is not None:
        if "*" in scope.bindings:
            break
        if name in scope.bindings:
            bindings = scope.bindings[name]
            if len(bindings) == 1:
                binding = bindings[0]
                node = binding.node
                if pos is None or node is None or (node.end_lineno, node.end_col_offset) <= pos:
                    return binding, scope
            break
        if scope.kind == "function":
            pos = None  # Free names are resolved when the function runs.
        scope = scope.parent
    return None, None


def _member(symbol):
    if isinstance(symbol, str) and symbol.startswith("polars."):
        module, _, name = symbol.rpartition(".")
        if name in _POLARS_MEMBERS.get(module, ()):
            return name
    return None


def _symbol(expr, scope, pos):
    if isinstance(expr, ast.Name):
        binding, owner = _binding(expr.id, scope, pos)
        if binding is not None:
            if isinstance(binding.value, str):
                return binding.value
            value = _symbol(binding.value, owner, _pos(binding.node)) if binding.value is not None else None
            if value is None and binding.annotation is not None:
                annotation = binding.annotation
                annotation_pos = _pos(annotation)
                if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
                    try:
                        annotation = ast.parse(annotation.value, mode="eval").body
                    except SyntaxError:
                        return None
                member = _member(_symbol(annotation, binding.annotation_scope, annotation_pos))
                return {"Expr": "series", "LazyFrame": "frame", "DataFrame": "eager"}.get(member)
            return value
    if isinstance(expr, ast.Attribute):
        root = _symbol(expr.value, scope, pos)
        if root in _POLARS_MEMBERS:
            return f"{root}.{expr.attr}"
    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Attribute):
            mode = _symbol(expr.func.value, scope, pos)
            name = expr.func.attr
            member = name if name in _POLARS_MEMBERS.get(mode, ()) else None
        else:
            mode = name = None
            member = _member(_symbol(expr.func, scope, pos))
        if member in _EXPR or member == "map_batches":
            return "series"
        if member in _LAZY:
            return "frame"
        if member == "DataFrame":
            return "eager"
        if mode == "eager" and name == "lazy":
            return "frame"
        if (mode == "frame" and name in _LAZY_METHODS) or (mode == "series" and name in _EXPR_METHODS):
            return mode
    return None


def _callback(call, index):
    kw = [item.value for item in call.keywords if item.arg == "function"]
    positional = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            break
        positional.append(arg)
    if len(positional) > index:
        return positional[index] if not kw else None
    return kw[0] if len(kw) == 1 else None


def _short(node):
    source = ast.unparse(node)
    return source if len(source) <= 80 else source[:77] + "..."


def _literal(node, value):
    return isinstance(node, ast.Constant) and node.value is value


def _target(scan, callback, scope, call, mode, path):
    if not isinstance(callback, ast.Name) or any(isinstance(arg, ast.Starred) for arg in call.args):
        return None
    if any(kw.arg is None or (kw.arg in _FLAGS.values() and kw.arg != _FLAGS[mode])
           or (mode == "series" and kw.arg == "returns_scalar" and not _literal(kw.value, False))
           for kw in call.keywords):
        return None
    binding, owner = _binding(callback.id, scope, _pos(call))
    fn = binding.value if owner is scan.module else None
    if not isinstance(fn, ast.FunctionDef) or fn.decorator_list or fn in scan.generators:
        return None
    args = fn.args
    positional = _positional(args)
    if (positional or args.vararg) and len(positional) - len(args.defaults) <= 1 and all(args.kw_defaults):
        return f"{path}:{callback.id}"
    return None


def discover(source: str, path: str) -> list[dict]:
    """Report literal batch promises in source order, without executing source."""
    scan = _Scan()
    scan.visit(ast.parse(source, filename=path))
    findings = []
    for call, scope, flags in sorted(scan.calls, key=lambda item: _pos(item[0])):
        func = call.func
        pos = _pos(call)
        if _member(_symbol(func, scope, pos)) == "map_batches":
            mode, receiver, index = "multi", getattr(func, "value", func), 1
        elif isinstance(func, ast.Attribute) and func.attr == "map_batches":
            receiver, index = func.value, 0
            symbol = _symbol(receiver, scope, pos)
            mode = symbol if symbol in {"series", "frame"} else "unknown"
        else:
            continue
        callback = _callback(call, index)
        kind = {ast.Name: "named", ast.Lambda: "lambda", ast.Attribute: "bound"}.get(type(callback), "other")
        for flag in flags:
            findings.append({
                "path": path, "line": call.lineno, "flag": flag, "receiver": _short(receiver),
                "confidence": "ambiguous" if mode == "unknown" else "polars", "kind": kind,
                "callback": _short(callback) if callback is not None else "<unknown>",
                "target": _target(scan, callback, scope, call, mode, path) if _FLAGS.get(mode) == flag else None,
                "mode": mode,
            })
    return findings
