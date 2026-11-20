"""Build a multi-line code-repair dataset from MBPP by injecting 2-3 bugs per instance.

Unlike build_code_dataset.py (single-line bugs), this creates instances where the
minimal fix requires changing 2 or 3 lines. This stresses DTE's iterative mask
expansion mechanism, which is designed for exactly this scenario.

Bug combinations:
  * two off-by-one errors in different functions
  * wrong operator + missing return
  * off-by-one + wrong operator
  * two missing returns (in different functions)

Output: data/code_repair_bugs_multiline.jsonl
Each record has:
  task_id, prompt, buggy_code, gold_code, bug_lines_1based (list), 
  bug_types (list), n_failing_tests, minimal_fix_lines (int)
"""
import json, ast, random, re, subprocess, sys, tempfile, os


def run_tests(code, test_list, timeout=8):
    """Run each test in the SAME namespace as the solution code."""
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
            if not p.stdout.strip() and p.returncode != 0:
                fails = len(test_list)
            return fails == 0, fails
        except Exception:
            return False, -1


# ---------------------------------------------------------------------------
# Single-bug injectors (same as build_code_dataset.py, but return all candidates)
# ---------------------------------------------------------------------------

def find_off_by_one_candidates(code):
    """Return list of (line_idx, new_line, bug_type) for off-by-one injections."""
    results = []
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("#"):
            continue
        # range(N) -> range(N-1)
        m = re.search(r"range\(\s*(\d+)\s*\)", s)
        if m:
            n = int(m.group(1))
            if n > 1:
                new = ln.replace("range(%d)" % n, "range(%d)" % (n - 1), 1)
                results.append((i, new, "off-by-one-range"))
        # comparison +/- 1
        m = re.search(r"(==|!=|<=|>=|<|>)\s*(\d+)", ln)
        if m:
            n = int(m.group(2))
            new = ln.replace("%s %d" % (m.group(1), n), "%s %d" % (m.group(1), n + 1), 1)
            results.append((i, new, "off-by-one-compare"))
    return results


def find_wrong_operator_candidates(code):
    """Return list of (line_idx, new_line, bug_type) for operator swaps."""
    results = []
    lines = code.split("\n")
    swaps = [("==", "!="), ("!=", "=="), ("<", ">"), (">", "<"), ("<=", ">="), (">=", "<=")]
    for i, ln in enumerate(lines):
        if ln.strip().startswith("#"):
            continue
        for old, new in swaps:
            if old in ln:
                results.append((i, ln.replace(old, new, 1), "wrong-operator"))
                break  # one swap per line
    return results


def find_missing_return_candidates(code):
    """Return list of (line_idx, new_line, bug_type) for missing-return injection."""
    results = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return results
    lines = code.split("\n")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.body:
            last = node.body[-1]
            if isinstance(last, ast.Return) and last.lineno is not None:
                li = last.lineno - 1
                if li < len(lines):
                    indent = lines[li][:len(lines[li]) - len(lines[li].lstrip())]
                    results.append((li, indent + "pass", "missing-return"))
    return results


# ---------------------------------------------------------------------------
# Multi-bug combination builder
# ---------------------------------------------------------------------------

def build_multiline_instance(gold, tests, seed):
    """Try to inject 2-3 bugs into `gold` such that the buggy code fails tests.
    
    Returns a dict or None if no valid multi-bug instance can be constructed.
    """
    rng = random.Random(seed)
    
    # Gather all candidate injections
    candidates = []
    for c in find_off_by_one_candidates(gold):
        candidates.append(c)
    for c in find_wrong_operator_candidates(gold):
        candidates.append(c)
    for c in find_missing_return_candidates(gold):
        candidates.append(c)
    
    if len(candidates) < 2:
        return None
    
    # We need at least 2 bugs on DIFFERENT lines
    candidates = [c for c in candidates]
    # Deduplicate by line (keep first candidate per line)
    seen_lines = set()
    unique_candidates = []
    for c in candidates:
        if c[0] not in seen_lines:
            seen_lines.add(c[0])
            unique_candidates.append(c)
    
    if len(unique_candidates) < 2:
        return None
    
    # Try combinations of 2, then 3
    from itertools import combinations
    for n_bugs in [2, 3]:
        if len(unique_candidates) < n_bugs:
            continue
        combo_list = list(combinations(range(len(unique_candidates)), n_bugs))
        rng.shuffle(combo_list)
        
        for combo in combo_list[:20]:  # try up to 20 random combos
            lines = gold.split("\n")
            bug_lines = []
            bug_types = []
            valid = True
            for ci in combo:
                li, new_line, btype = unique_candidates[ci]
                if li >= len(lines):
                    valid = False
                    break
                lines[li] = new_line
                bug_lines.append(li + 1)  # 1-based
                bug_types.append(btype)
            
            if not valid:
                continue
            
            buggy = "\n".join(lines)
            bok, bfails = run_tests(buggy, tests)
            if not bok and bfails > 0:
                # Verify the gold passes (sanity check)
                gok, _ = run_tests(gold, tests)
                if not gok:
                    continue
                
                # Compute minimal fix lines (diff between buggy and gold)
                import difflib
                sm = difflib.SequenceMatcher(
                    a=buggy.splitlines(), b=gold.splitlines())
                n_fix_lines = 0
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag != "equal":
                        n_fix_lines += max(i2 - i1, j2 - j1)
                
                return {
                    "buggy_code": buggy,
                    "gold_code": gold,
                    "bug_lines_1based": sorted(bug_lines),
                    "bug_types": bug_types,
                    "n_failing_tests": bfails,
                    "minimal_fix_lines": n_fix_lines,
                }
    return None


def build_multiline_dataset(n_target=20, seed=42):
    """Build a dataset of multi-line bug instances from MBPP."""
    random.seed(seed)
    with open("data/mbpp_test.jsonl") as f:
        problems = [json.loads(l) for l in f]
    random.shuffle(problems)
    
    dataset = []
    used_tasks = set()
    for prob in problems:
        if len(dataset) >= n_target:
            break
        if prob["task_id"] in used_tasks:
            continue
        
        gold = prob["code"]
        tests = prob["test_list"]
        
        # Verify gold passes first
        gok, _ = run_tests(gold, tests)
        if not gok:
            continue
        
        result = build_multiline_instance(gold, tests, seed + len(dataset))
        if result is None:
            continue
        
        result["task_id"] = prob["task_id"]
        result["prompt"] = prob["text"]
        dataset.append(result)
        used_tasks.add(prob["task_id"])
    
    return dataset


if __name__ == "__main__":
    ds = build_multiline_dataset(n_target=20, seed=42)
    with open("data/code_repair_bugs_multiline.jsonl", "w") as f:
        for r in ds:
            f.write(json.dumps(r) + "\n")
    
    from collections import Counter
    print(f"Built {len(ds)} multi-line buggy-code instances")
    type_counts = Counter()
    for r in ds:
        type_counts[tuple(sorted(r["bug_types"]))] += 1
    print("Bug type combinations:")
    for combo, count in type_counts.most_common():
        print(f"  {combo}: {count}")
    print(f"Minimal fix lines distribution: "
          f"{dict(Counter(r['minimal_fix_lines'] for r in ds))}")

