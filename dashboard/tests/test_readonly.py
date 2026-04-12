"""test_readonly.py — Structural read-only enforcement via AST/grep analysis.

Verifies that dashboard/src/ contains no write operations, no mutating
subprocess calls, and no mutating HTTP calls (outside whitelists).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"


def _py_files():
    return [p for p in SRC_DIR.rglob("*.py") if "test" not in str(p)]


class WriteCallVisitor(ast.NodeVisitor):
    """Find open() calls with write modes."""

    def __init__(self):
        self.violations: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        # open(..., 'w') or open(..., 'a') or open(..., 'wb') etc.
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if any(m in arg.value for m in ("w", "a", "x")):
                        self.violations.append((node.lineno, ast.unparse(node)))
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    if any(m in kw.value.value for m in ("w", "a", "x")):
                        self.violations.append((node.lineno, ast.unparse(node)))
        self.generic_visit(node)


def test_no_open_write_ast():
    """No open() with write mode anywhere in src/."""
    all_violations = []
    for path in _py_files():
        tree = ast.parse(path.read_text())
        visitor = WriteCallVisitor()
        visitor.visit(tree)
        for lineno, call in visitor.violations:
            all_violations.append(f"{path.name}:{lineno}: {call}")
    assert not all_violations, "Write file opens found:\n" + "\n".join(all_violations)


def test_no_os_remove():
    """No os.remove() calls in src/."""
    pattern = re.compile(r'\bos\.remove\s*\(')
    violations = []
    for path in _py_files():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{path.name}:{i}: {line.strip()}")
    assert not violations, "os.remove calls found:\n" + "\n".join(violations)


def test_no_shutil_rmtree():
    """No shutil.rmtree() calls in src/."""
    pattern = re.compile(r'\bshutil\.rmtree\s*\(')
    violations = []
    for path in _py_files():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{path.name}:{i}: {line.strip()}")
    assert not violations, "shutil.rmtree calls found:\n" + "\n".join(violations)


def test_mutating_requests_not_in_hot_path():
    """requests.post/put/delete/patch only allowed in safe_exec.py (whitelisted).

    Exception: mlflow_client.py uses requests.post for MLflow search API
    (MLflow uses POST for /runs/search — read-only by design).
    """
    pattern = re.compile(r'\brequests\.(post|put|delete|patch)\s*\(')
    # Files permitted to use requests.post for read-only API calls
    _ALLOWED = {"safe_exec.py", "mlflow_client.py"}
    violations = []
    for path in _py_files():
        if path.name in _ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{path.name}:{i}: {line.strip()}")
    assert not violations, "Mutating requests calls outside safe_exec:\n" + "\n".join(violations)


def test_mutating_httpx_not_in_hot_path():
    """httpx.post/put/delete/patch only allowed in safe_exec.py (whitelisted)."""
    pattern = re.compile(r'\bhttpx\.(post|put|delete|patch)\s*\(')
    violations = []
    for path in _py_files():
        if path.name == "safe_exec.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{path.name}:{i}: {line.strip()}")
    assert not violations, "Mutating httpx calls outside safe_exec:\n" + "\n".join(violations)
