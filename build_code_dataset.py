"""Build a realistic code-repair dataset from MBPP by injecting single-line bugs.

For each selected MBPP problem:
  1. Verify the gold solution passes its tests.
  2. Inject ONE single-line bug (off-by-one, wrong operator, missing return).
  3. Verify the buggy version FAILS at least one test.
  4. Record (buggy_code, gold_code, bug_line, bug_type) — ground-truth minimal fix = 1 line.

Output: data/code_repair_bugs.jsonl
"""
import json, ast, random, re, subprocess, sys, tempfile, os

def run_tests(code, test_list, timeout=8):
    """Run each test in the SAME namespace as the solution code.
    Returns (all_passed, n_fail)."""
    with tempfile.TemporaryDirectory() as d:
        tf = os.path.join(d, "test.py")
        tc = code + "\n"
        for t in test_list:
            tc += "try:\n    %s\n    print('PASS')\nexcept Exception as e:\n    print('FAIL', repr(e))\n" % t
        with open(tf, "w") as f:
            f.write(tc)
        try:
            p = subprocess.run([sys.executable, tf], capture_output=True, text=True, timeout=timeout)
            fails = len(re.findall(r"^FAIL", p.stdout, re.MULTILINE))
            return fails == 0, fails
        except Exception:
            return False, -1

def inject_off_by_one(code):
    """Change a range bound or +/-1 in a loop/condition."""
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        s = ln.strip()
        # range(N) -> range(N-1) style: find 'range(' with an int or simple expr
        m = re.search(r"range\(\s*(\d+)\s*\)", s)
        if m:
            n = int(m.group(1))
            if n > 1:
                new = ln.replace("range(%d)" % n, "range(%d)" % (n - 1), 1)
                return i, new, "off-by-one-range"
    # fallback: +/- 1 on a numeric literal in a comparison
    for i, ln in enumerate(lines):
        m = re.search(r"(==|!=|<=|>=|<|>)\s*(\d+)", ln)
        if m:
            n = int(m.group(2))
            new = ln.replace("%s %d" % (m.group(1), n), "%s %d" % (m.group(1), n + 1), 1)
            return i, new, "off-by-one-compare"
    return None

def inject_wrong_operator(code):
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        for old, new in [("==", "!="), ("!=", "=="), ("<", ">"), (">", "<"), ("<=", ">="), (">=", "<=")]:
            if old in ln and not ln.strip().startswith("#"):
                return i, ln.replace(old, new, 1), "wrong-operator"
    return None

def inject_missing_return(code):
    """Replace the final 'return X' of a function with 'pass' so it returns None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.body:
            last = node.body[-1]
            if isinstance(last, ast.Return) and last.lineno is not None:
                li = last.lineno - 1
                lines = code.split("\n")
                if li < len(lines):
                    indent = lines[li][:len(lines[li]) - len(lines[li].lstrip())]
                    return li, indent + "pass", "missing-return"
    return None

def build_dataset(n_target=30, seed=42):
    random.seed(seed)
    with open("data/mbpp_test.jsonl") as f:
        problems = [json.loads(l) for l in f]
    random.shuffle(problems)

    injectors = [inject_off_by_one, inject_wrong_operator, inject_missing_return]
    dataset = []
    for prob in problems:
        if len(dataset) >= n_target:
            break
        gold = prob["code"]
        tests = prob["test_list"]
        # verify gold passes
        gok, _ = run_tests(gold, tests)
        if not gok:
            continue
        # try each injector until one produces a failing buggy version
        for inj in injectors:
            res = inj(gold)
            if res is None:
                continue
            li, new_line, btype = res
            lines = gold.split("\n")
            if li >= len(lines):
                continue
            lines[li] = new_line
            buggy = "\n".join(lines)
            bok, bfails = run_tests(buggy, tests)
            if not bok and bfails > 0:
                dataset.append({
                    "task_id": prob["task_id"],
                    "prompt": prob["text"],
                    "buggy_code": buggy,
                    "gold_code": gold,
                    "bug_line_1based": li + 1,
                    "bug_type": btype,
                    "n_failing_tests": bfails,
                })
                break
    return dataset

def build_balanced(per_type=10, seed=42):
    random.seed(seed)
    with open("data/mbpp_test.jsonl") as f:
        problems = [json.loads(l) for l in f]
    random.shuffle(problems)
    injectors = {"off-by-one-compare": inject_off_by_one,
                 "wrong-operator": inject_wrong_operator,
                 "missing-return": inject_missing_return}
    by_type = {k: [] for k in injectors}
    used_tasks = set()
    for prob in problems:
        if all(len(v) >= per_type for v in by_type.values()):
            break
        gold = prob["code"]
        tests = prob["test_list"]
        gok, _ = run_tests(gold, tests)
        if not gok or prob["task_id"] in used_tasks:
            continue
        for btype, inj in injectors.items():
            if len(by_type[btype]) >= per_type:
                continue
            res = inj(gold)
            if res is None:
                continue
            li, new_line, _ = res
            lines = gold.split("\n")
            if li >= len(lines):
                continue
            lines[li] = new_line
            buggy = "\n".join(lines)
            bok, bfails = run_tests(buggy, tests)
            if not bok and bfails > 0:
                by_type[btype].append({
                    "task_id": prob["task_id"], "prompt": prob["text"],
                    "buggy_code": buggy, "gold_code": gold,
                    "bug_line_1based": li + 1, "bug_type": btype,
                    "n_failing_tests": bfails,
                })
                used_tasks.add(prob["task_id"])
                break
    out = [r for v in by_type.values() for r in v]
    random.shuffle(out)
    return out

if __name__ == "__main__":
    ds = build_balanced(per_type=10)
    with open("data/code_repair_bugs.jsonl", "w") as f:
        for r in ds:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"Built {len(ds)} buggy-code instances (balanced)")
    print("By bug type:", dict(Counter(r['bug_type'] for r in ds)))
