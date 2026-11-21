"""Recover benchmark inputs from versioned sources and select a study split."""

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
from pathlib import Path
import pickle
import re
import shutil
import tempfile
from urllib.request import Request, urlopen


def table(path, delimiter="\t"):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_table(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources(data, raw, dataset, fetch):
    files = {}
    for entry in table(data / "sources.tsv"):
        if entry["dataset"] != dataset:
            continue
        path = raw / entry["filename"]
        if not path.exists() and fetch:
            raw.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=raw, delete=False) as stream:
                temp = Path(stream.name)
            try:
                request = Request(entry["url"], headers={"User-Agent": "TRACE-DTA data retrieval"})
                with urlopen(request, timeout=120) as resp, temp.open("wb") as stream:
                    shutil.copyfileobj(resp, stream)
                if sha256(temp) != entry["sha256"]:
                    raise ValueError(
                        f"Downloaded file differs from the study snapshot: {path.name}"
                    )
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Supply the source file listed in sources.tsv: {path}")
        files[path.name] = path
    return files


def pkd(nanomolar):
    with localcontext() as ctx:
        ctx.prec = 50
        value = Decimal(str(nanomolar))
        if not value.is_finite() or value <= 0:
            raise ValueError("Expected a positive dissociation constant")
        return float((Decimal(9) - value.log10()).quantize(Decimal("0.000000000001")))


def davis_records(index, files):
    # The original authors distributed these four files as pickles. Verify each
    # file against the published input manifest before loading it.
    with files["Y"].open("rb") as f:
        matrix = pickle.load(f, encoding="latin1")
    members = {}
    for filename in {r["member_file"] for r in index}:
        with files[filename].open("rb") as f:
            members[filename] = pickle.load(f, encoding="latin1")
    for r in index:
        drug, sm, target, seq, _ = members[r["member_file"]][int(r["member_row"])]
        if (str(drug), str(target)) != (r["drug_id"], r["target_id"]):
            raise ValueError("Davis member identity mismatch")
        values = {matrix[int(r["matrix_row"]), int(col)] for col in r["matrix_columns"].split(",")}
        if len(values) != 1:
            raise ValueError("Davis matched records differ in Kd")
        yield r["sample"], (sm, seq, pkd(values.pop()))


def bindingdb_records(index, files):
    with files["bindingdb_kd.tab"].open(newline="") as f:
        measurements = {line: row for line, row in enumerate(csv.DictReader(f, delimiter="\t"), 2)}
    for r in index:
        matches = [measurements[int(i)] for i in r["tdc_lines"].split(",")]
        first = matches[0]
        if any(
            (row["ID1"], row["ID2"]) != (r["drug_id"], r["target_id"])
            or (row["X1"], row["X2"], Decimal(row["Y"]))
            != (first["X1"], first["X2"], Decimal(first["Y"]))
            for row in matches
        ):
            raise ValueError("BindingDB measurements differ within a matched group")
        yield r["sample"], (first["X1"], first["X2"], pkd(first["Y"]))


def pdbbind_records(index, files, data):
    source = {r[""]: r for r in table(files["LP_PDBBind.csv"], delimiter=",")}
    fixes = {r["pdb_id"]: r["smiles"] for r in table(data / "records/pdbbind_ligands.tsv")}
    exponent = {"M": 9, "mM": 6, "uM": 3, "µM": 3, "nM": 0, "pM": -3, "fM": -6}
    for r in index:
        pid = r["sample"]
        row = source[pid]
        match = re.fullmatch(
            r"Kd=((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Zµ]+)", row["kd/ki"]
        )
        if not match:
            raise ValueError(f"Expected an exact Kd for {pid}")
        magnitude = Decimal(match[1]).scaleb(exponent[match[2]])
        yield pid, (fixes.get(pid, row["smiles"]), row["seq"], pkd(magnitude))


def restore(data, raw, dataset, fetch=False):
    index = table(data / "records" / (dataset + ".tsv"))
    files = sources(data, raw, dataset, fetch)
    if dataset == "davis":
        records = davis_records(index, files)
    elif dataset == "bindingdb":
        records = bindingdb_records(index, files)
    else:
        records = pdbbind_records(index, files, data)
    return dict(records)


def kinase_panel(data, raw, output, fetch):
    records = restore(data, raw, "davis", fetch)
    targets = {r["target_id"]: records[r["sample"]][1] for r in table(data / "records/davis.tsv")}
    assays = table(data / "records/kinase_panel_targets.tsv")
    compounds = table(data / "records/kinase_panel_compounds.tsv")
    pairs = []
    for assay in assays:
        sequence = targets[assay["davis_target"]]
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != assay["sequence_sha256"]:
            raise ValueError("Kinase panel sequence identity mismatch")
        for drug in compounds:
            pairs.append((assay["assay"] + "|" + drug["compound"], drug["smiles"], sequence))
    write_table(output / "pairs.tsv", ("sample", "smiles", "chains"), pairs)


def make_split(args, out):
    dataset, data = args.dataset, args.data
    records = restore(data, args.raw, dataset, args.fetch)
    name = "cold_target.tsv" if args.protocol == "cold_target" else "sequence_clusters.tsv"
    members = {key: [] for key in ("train", "validation", "test")}
    for r in table(data / "splits" / name):
        if r["dataset"] == dataset and r["fold"] in (str(args.fold), "all"):
            members[r["subset"]].append(r["sample"])
    if any(not x for x in members.values()):
        raise ValueError(f"Fold {args.fold} is unavailable for {dataset} under {args.protocol}")
    pool = members["train"]
    if args.protocol == "label_budget" and args.percent != 100:
        selected = [
            r["sample"]
            for r in table(data / "splits/label_budgets.tsv")
            if r["dataset"] == dataset
            and int(r["percent"]) == args.percent
            and int(r["seed"]) == args.seed
        ]
        if not selected or not set(selected) <= set(pool):
            raise ValueError("Label-budget membership does not match the outer training set")
        members["train"] = selected
    fields = ("sample", "smiles", "chains", "pkd")
    for role, ids in members.items():
        write_table(out / (role + ".tsv"), fields, ((i, *records[i]) for i in ids))
    write_table(out / "pool.tsv", fields[:3], ((i, *records[i][:2]) for i in pool))
    write_table(
        out / "proteins.tsv", ("chains",), ((s,) for s in sorted({r[1] for r in records.values()}))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("davis", "bindingdb", "pdbbind", "kinase_panel"), required=True
    )
    parser.add_argument(
        "--protocol",
        choices=("cold_target", "sequence_cluster", "label_budget"),
        default="cold_target",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--percent", type=int, choices=(10, 25, 50, 100), default=100)
    parser.add_argument(
        "--seed", type=int, choices=(1, 7, 42), default=1, help="Label-budget subset seed"
    )
    parser.add_argument("--fetch", action="store_true", help="Download missing upstream files")
    args = parser.parse_args()
    if args.protocol == "label_budget" and (
        args.dataset not in {"davis", "bindingdb"} or args.fold != 1
    ):
        parser.error("Label budgets use sequence-cluster fold 1 of Davis or BindingDB")
    if args.percent != 100 and args.protocol != "label_budget":
        parser.error("Use --protocol label_budget to change the training fraction")
    if args.output.exists():
        parser.error("Choose a new output directory for this split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        out = Path(temp) / "prepared"
        out.mkdir()
        if args.dataset == "kinase_panel":
            kinase_panel(args.data, args.raw, out, args.fetch)
        else:
            make_split(args, out)
        out.rename(args.output)


if __name__ == "__main__":
    main()
