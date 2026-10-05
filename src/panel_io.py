"""The one entry point for reading the built panel.

    from src.panel_io import load_panel
    fit = load_panel(roles=["fit"], columns=["loan_id", "run_length", "next_state"])

Every call first validates the WHOLE panel on disk against panel_manifest.json,
which src/panel.py writes only at the end of a successful build: partition
count, filenames, per-partition row counts, column names, Arrow dtypes, and the
build stamp in every partition's footer. All of it is read from parquet footers;
no data is read to validate. Any disagreement raises PanelError naming exactly
what disagreed. There is no override and no warning path - a mismatch means
downstream numbers would be wrong.

Only then is the requested subset loaded. The two row filters differ:
    quarters=   selects FILES: only those partitions are opened.
    roles=      selects ROWS: role labels occur in every file, so every
                selected file is read and filtered as it is read.
columns= projects at read time. Partitions are read one at a time and only the
requested columns and rows are kept, so a subset never materialises the panel.
"""

import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from src.panel import (CATEGORICAL_FEATURES, MANIFEST_NAME, PANEL_DIR,
                       PANEL_FILE, ROLES, STAMP_KEY)


class PanelError(RuntimeError):
    """The panel on disk cannot be trusted, or the request cannot be served."""


def read_manifest():
    path = os.path.join(PANEL_DIR, MANIFEST_NAME)
    if not os.path.exists(path):
        raise PanelError("no manifest at %s - the panel must be rebuilt "
                         "(python -m src.panel)" % path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate(manifest):
    """Check every partition on disk against the manifest; raise on any issue."""
    problems = []
    listed = {p["file"]: p for p in manifest["partitions"]}
    if manifest["n_partitions"] != len(listed):
        problems.append("manifest says %d partitions but lists %d"
                        % (manifest["n_partitions"], len(listed)))
    on_disk = {os.path.basename(p) for p in
               glob.glob(os.path.join(PANEL_DIR, PANEL_FILE % "*"))}
    if len(on_disk) != len(listed):
        problems.append("%d partitions on disk, manifest lists %d"
                        % (len(on_disk), len(listed)))
    for name in sorted(set(listed) - on_disk):
        problems.append("%s: listed in the manifest, missing on disk" % name)
    for name in sorted(on_disk - set(listed)):
        problems.append("%s: on disk, not in the manifest" % name)

    want = [(c["name"], c["type"]) for c in manifest["schema"]]
    # Non-numeric features must carry exactly the type src.panel declares for
    # them, whatever the manifest says: the manifest records what was written,
    # it does not get to redefine what a feature is.
    want_types = dict(want)
    for col, declared in CATEGORICAL_FEATURES.items():
        if want_types.get(col) != declared:
            problems.append("column %s: manifest records %s, the schema "
                            "declares %s" % (col, want_types.get(col), declared))
    for name in sorted(set(listed) & on_disk):
        with pq.ParquetFile(os.path.join(PANEL_DIR, name)) as f:
            rows = f.metadata.num_rows
            schema = f.schema_arrow
        if rows != listed[name]["rows"]:
            problems.append("%s: %d rows on disk, manifest says %d"
                            % (name, rows, listed[name]["rows"]))
        got = [(fld.name, str(fld.type)) for fld in schema]
        if [c for c, _ in got] != [c for c, _ in want]:
            problems.append("%s: columns %s, manifest says %s"
                            % (name, [c for c, _ in got], [c for c, _ in want]))
        else:
            for (col, t_got), (_, t_want) in zip(got, want):
                if t_got != t_want:
                    problems.append("%s: column %s is %s, manifest says %s"
                                    % (name, col, t_got, t_want))
        stamp = (schema.metadata or {}).get(STAMP_KEY)
        stamp = stamp.decode("ascii") if stamp is not None else None
        if stamp != manifest["build_stamp"]:
            problems.append("%s: build stamp %r, manifest says %r"
                            % (name, stamp, manifest["build_stamp"]))
    if problems:
        raise PanelError("panel on disk does not match %s:\n    %s"
                         % (MANIFEST_NAME, "\n    ".join(problems)))


def load_panel(roles=None, columns=None, quarters=None):
    """Validate the whole panel, then return the requested subset as a DataFrame.

    roles     role names to keep (fit, condition, evaluate); None = all rows
    columns   columns to return, in this order; None = all
    quarters  quarter names (e.g. "2006Q1") whose partitions to read; None = all
    """
    manifest = read_manifest()
    validate(manifest)

    names = [c["name"] for c in manifest["schema"]]
    cols = names if columns is None else list(columns)
    unknown = [c for c in cols if c not in names]
    if unknown:
        raise PanelError("unknown column(s) %s; the panel has %s"
                         % (unknown, names))
    by_quarter = {p["quarter"]: p for p in manifest["partitions"]}
    if quarters is None:
        chosen = [p["quarter"] for p in manifest["partitions"]]
    else:
        chosen = list(quarters)
        missing = [q for q in chosen if q not in by_quarter]
        if missing:
            raise PanelError("quarter(s) %s not in the panel; it has %s"
                             % (missing, sorted(by_quarter)))
    if not chosen:
        raise PanelError("no quarters requested")
    filters = None
    read_cols = cols
    if roles is not None:
        roles = list(roles)
        bad = [r for r in roles if r not in ROLES]
        if bad:
            raise PanelError("unknown role(s) %s; roles are %s" % (bad, ROLES))
        filters = [("role", "in", roles)]
        if "role" not in cols:
            read_cols = cols + ["role"]

    tables = []
    for q in chosen:
        t = pq.read_table(os.path.join(PANEL_DIR, by_quarter[q]["file"]),
                          columns=read_cols, filters=filters)
        tables.append(t.select(cols))
    return pa.concat_tables(tables).to_pandas()
