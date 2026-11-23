from textwrap import dedent

import pytest

from batchseam.discovery import discover


def scan(source):
    return discover(dedent(source).lstrip(), "workflow.py")


def test_only_explicit_true_is_reported_with_call_location():
    assert scan('''
        import polars as p
        def transform(batch):
            return batch
        p.col("value").map_batches(
            transform, is_elementwise=True,
        )
        p.col("value").map_batches(transform, is_elementwise=1)
        p.col("value").map_batches(transform, is_elementwise=False)
        p.col("value").map_batches(transform, is_elementwise=enabled)
        p.col("value").map_batches(transform, **{"is_elementwise": True})
        sample = 'p.col("value").map_batches(transform, is_elementwise=True)'
    ''') == [{
        "path": "workflow.py", "line": 4, "flag": "is_elementwise", "receiver": "p.col('value')",
        "confidence": "polars", "kind": "named", "callback": "transform", "target": "workflow.py:transform",
        "mode": "series",
    }]


def test_receivers_from_imports_annotations_and_constructors():
    rows = scan('''
        import polars
        from polars import Expr as Expression, LazyFrame as LF, struct as struct_expr
        def fn(batch, *, optional=True):
            return batch
        def build(expr: Expression, plan: "LF"):
            expr.map_batches(fn, is_elementwise=True)
            plan.map_batches(function=fn, streamable=True)
        selected = struct_expr("x", "y").alias("xy")
        selected.map_batches(fn, is_elementwise=True)
        LF().filter(polars.col("x") > 0).map_batches(fn, streamable=True)
        polars.DataFrame().lazy().map_batches(fn, streamable=True)
        polars.scan_parquet("input.parquet").map_batches(fn, streamable=True)
    ''')
    assert [row["mode"] for row in rows] == ["series", "frame", "series", "frame", "frame", "frame"]
    assert {row["confidence"] for row in rows} == {"polars"}
    assert {row["target"] for row in rows} == {"workflow.py:fn"}


def test_multi_input_callback_position_and_keyword():
    rows = scan('''
        import polars as p
        from polars import map_batches as apply_batches
        def combine(columns):
            return columns[0]
        p.map_batches(["a", "b"], combine, is_elementwise=True)
        p.map_batches(function=combine, exprs=["a", "b"], is_elementwise=True)
        apply_batches(["a", "b"], lambda columns: columns[0], is_elementwise=True)
        apply_batches(["a", "b"], registry.combine, is_elementwise=True)
    ''')
    assert [(row["receiver"], row["callback"], row["kind"], row["target"]) for row in rows] == [
        ("p", "combine", "named", "workflow.py:combine"),
        ("p", "combine", "named", "workflow.py:combine"),
        ("apply_batches", "lambda columns: columns[0]", "lambda", None),
        ("apply_batches", "registry.combine", "bound", None),
    ]
    assert {row["mode"] for row in rows} == {"multi"}


def test_unrelated_receivers_and_shadowed_imports_are_ambiguous():
    rows = scan('''
        import unrelated as polars
        from unrelated import col, map_batches
        import polars as p
        def fn(batch):
            return batch
        def shadow(p):
            p.col("x").map_batches(fn, is_elementwise=True)
        polars.col("x").map_batches(fn, is_elementwise=True)
        col("x").map_batches(fn, is_elementwise=True)
        obj.map_batches(fn, streamable=True, is_elementwise=True)
        map_batches(["x"], fn, is_elementwise=True)
        def overwritten():
            p.col("x").map_batches(fn, is_elementwise=True)
            p = unrelated
    ''')
    assert len(rows) == 6
    assert [(row["flag"], row["receiver"]) for row in rows[3:5]] == [
        ("streamable", "obj"), ("is_elementwise", "obj"),
    ]
    assert {(row["confidence"], row["mode"], row["target"]) for row in rows} == {("ambiguous", "unknown", None)}


def test_only_unshadowed_usable_module_functions_get_targets():
    rows = scan('''
        import polars as p
        from helpers import imported
        def fn(batch):
            return batch
        def same(batch):
            return batch
        same = imported
        def bad_arity(batch, config):
            return batch
        def needs_keyword(batch, *, config):
            return batch
        def generator(batch):
            yield batch
        async def async_fn(batch):
            return batch
        @decorator
        def decorated(batch):
            return batch
        if enabled:
            def conditional(batch):
                return batch
        def outer(fn):
            def nested(batch):
                return batch
            p.col("x").map_batches(fn, is_elementwise=True)
            p.col("x").map_batches(nested, is_elementwise=True)
        class C:
            def fn(self, batch):
                return batch
            direct = p.col("x").map_batches(fn, is_elementwise=True)
            def build(self):
                return p.col("x").map_batches(fn, is_elementwise=True)
        [p.col("x").map_batches(fn, is_elementwise=True) for fn in callbacks]
        p.col("x").map_batches(same, is_elementwise=True)
        p.col("x").map_batches(bad_arity, is_elementwise=True)
        p.col("x").map_batches(needs_keyword, is_elementwise=True)
        p.col("x").map_batches(generator, is_elementwise=True)
        p.col("x").map_batches(async_fn, is_elementwise=True)
        p.col("x").map_batches(decorated, is_elementwise=True)
        p.col("x").map_batches(conditional, is_elementwise=True)
        p.col("x").map_batches(imported, is_elementwise=True)
        p.col("x").map_batches(factory(), is_elementwise=True)
    ''')
    assert len(rows) == 14
    assert [row["target"] for row in rows] == [None] * 3 + ["workflow.py:fn"] + [None] * 10
    assert rows[-1]["kind"] == "other"
    assert rows[-1]["callback"] == "factory()"


def test_missing_or_dynamic_callback_and_incompatible_flag_have_no_target():
    rows = scan('''
        import polars as pl
        def fn(batch, /):
            return batch
        pl.col("x").map_batches(is_elementwise=True)
        pl.col("x").map_batches(*callbacks, is_elementwise=True)
        pl.col("x").map_batches(fn, is_elementwise=True, **opts)
        pl.col("x").map_batches(fn, is_elementwise=True, returns_scalar=True)
        pl.col("x").map_batches(fn, streamable=True)
        pl.col("x").map_batches(fn, function=other, is_elementwise=True)
    ''')
    assert [row["callback"] for row in rows] == ["<unknown>", "<unknown>", "fn", "fn", "fn", "<unknown>"]
    assert all(row["target"] is None for row in rows)


def test_explicit_global_reassignment_blocks_a_target():
    rows = scan('''
        import polars as p
        def fn(batch):
            return batch
        def replace():
            global fn
            fn = some_other_callback
        replace()
        p.col("x").map_batches(fn, is_elementwise=True)
    ''')
    assert rows[0]["confidence"] == "polars"
    assert rows[0]["target"] is None


def test_discovery_does_not_execute_source_and_exposes_syntax_errors():
    rows = scan('''
        raise RuntimeError("must not execute")
        x.map_batches(f, streamable=True)
    ''')
    assert rows[0]["line"] == 2
    with pytest.raises(SyntaxError) as exc:
        discover("def bad(", "broken.py")
    assert exc.value.filename == "broken.py"


def test_receiver_in_long_fluent_pipeline():
    chain = "pl.LazyFrame()" + ".clone()" * 35
    row, = discover(f"import polars as pl\n{chain}.map_batches(fn, streamable=True)", "pipeline.py")
    assert (row["mode"], row["confidence"]) == ("frame", "polars")


def test_optional_import_and_renamed_methods_select_the_actual_callback():
    rows = scan('''
        try:
            import polars as p
            from polars import map_batches as many
        except ImportError:
            p = many = None
        def ratio(cols): return cols[0] / cols[1]
        def doubled(s): return s * 2
        p.map_batches(["x", "y"], ratio, is_elementwise=True)
        many(["x", "y"], ratio, is_elementwise=True)
        def plan(expr: p.Expr):
            apply = expr.map_batches
            return apply(doubled, is_elementwise=True)
    ''')
    assert [(r["mode"], r["callback"]) for r in rows] == [
        ("multi", "ratio"), ("multi", "ratio"), ("series", "doubled")]
    assert rows[-1]["target"] == "workflow.py:doubled"


def test_nonlocal_rebinding_makes_enclosing_receiver_ambiguous():
    row, = scan('''
        import polars as pl
        def build():
            obj = pl.LazyFrame()
            def set_expr():
                nonlocal obj
                obj = pl.col("x")
            set_expr()
            return obj.map_batches(lambda s: s + 1, is_elementwise=True)
    ''')
    assert (row["confidence"], row["mode"], row["target"]) == ("ambiguous", "unknown", None)
