#!/usr/bin/env python3
"""C5 native OmniFlow dependency gate: static, not proof of runtime isolation."""
from __future__ import annotations
import ast
from pathlib import Path
import sys

BAD_PREFIXES=("comfy","comfy_api","comfyui","custom_nodes","folder_paths","comfy_extras")

def bad_name(name:str)->bool:
    return any(name==prefix or name.startswith(prefix+".") for prefix in BAD_PREFIXES)

def scan_python(source:str)->list[str]:
    tree=ast.parse(source)
    issues=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for alias in node.names:
                if bad_name(alias.name):
                    issues.append(f"import {alias.name} at line {node.lineno}")
        elif isinstance(node,ast.ImportFrom):
            if node.module and bad_name(node.module):
                issues.append(f"from {node.module} at line {node.lineno}")
        elif isinstance(node,ast.Call):
            func=node.func
            dynamic=(isinstance(func,ast.Name) and func.id=="__import__" or
                     isinstance(func,ast.Attribute) and func.attr=="import_module")
            if dynamic and node.args and isinstance(node.args[0],ast.Constant):
                target=node.args[0].value
                if isinstance(target,str) and bad_name(target):
                    issues.append(f"dynamic import {target} at line {node.lineno}")
    return issues

def main()->int:
    root=Path(__file__).resolve().parents[1]
    issues=[]
    for native in ("omniflow-control","omniflow-worker","omniflow-ui"):
        location=root/native
        if not location.exists(): continue
        for path in location.rglob("*.py"):
            if path.name.startswith(".") or "__pycache__" in path.parts: continue
            try:
                found=scan_python(path.read_text(encoding="utf-8"))
                issues.extend(f"{path.relative_to(root)}: {issue}" for issue in found)
            except SyntaxError as exc:
                issues.append(f"{path.relative_to(root)}: invalid Python at {exc.lineno}")
        for name in ("Cargo.toml","pyproject.toml","requirements.txt"):
            for manifest in location.rglob(name):
                for n,line in enumerate(manifest.read_text(encoding="utf-8").splitlines(),1):
                    trimmed=line.strip().split("#",1)[0].strip().lower()
                    if any(trimmed.startswith(prefix) for prefix in BAD_PREFIXES):
                        issues.append(f"{manifest.relative_to(root)}:{n}: native dependency {trimmed}")
    for issue in issues:
        print("C5 FAIL:",issue,file=sys.stderr)
    if issues: return 1
    print("C5 PASS: no identified ComfyUI direct/dynamic imports in native OmniFlow roots")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
