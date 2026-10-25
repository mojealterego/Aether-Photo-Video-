"""Fail-closed validation of a ComfyUI API graph against live server metadata.

The server's /object_info/<class> exposes required inputs, available output
ports and widget choices. Those choices include installed model checkpoints
for arbitrary extensions (CodeFormer, ControlNet, Depth Anything, etc.).
No model is assumed installed and validation does not execute GPU inference.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote


def verify_remote_graph(client: Any, graph: dict[str, dict]) -> dict:
    if not isinstance(graph, dict) or not graph:
        raise ValueError("Empty ComfyUI graph")
    if not all(isinstance(uid, str) and uid.isdecimal() and
               isinstance(node, dict) for uid, node in graph.items()):
        raise ValueError("ComfyUI graph needs numeric node IDs and node objects")

    metadata: dict[str, dict] = {}
    for node in graph.values():
        name = node.get("class_type")
        if not isinstance(name, str) or not name or "/" in name or len(name) > 256:
            raise ValueError("Bad ComfyUI node class name")
        if name in metadata:
            continue
        body = client.json("/object_info/" + quote(name, safe=""))
        if not isinstance(body, dict) or not isinstance(body.get(name), dict):
            raise RuntimeError(f"Missing ComfyUI node class on server: {name}")
        info = body[name]
        if not isinstance(info.get("input"), dict) or not isinstance(info.get("output"), list):
            raise RuntimeError(f"Invalid remote ComfyUI node metadata for {name}")
        for group in ("required", "optional", "hidden"):
            fields = info["input"].get(group, {})
            if not isinstance(fields, dict):
                raise RuntimeError(f"Invalid remote ComfyUI {group} metadata: {name}")
        metadata[name] = info

    for uid, node in graph.items():
        name = node["class_type"]
        allowed = metadata[name]["input"]
        required = allowed.get("required", {})
        optional = allowed.get("optional", {})
        hidden = allowed.get("hidden", {})
        accepted = set(required) | set(optional) | set(hidden)
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or not set(inputs).issubset(accepted):
            raise RuntimeError(f"Incompatible remote node input schema: {name} ({uid})")
        for key in required:
            if key not in inputs:
                raise RuntimeError(f"Missing required ComfyUI input {name}.{key}")
        for key, field in inputs.items():
            descriptor = required.get(key, optional.get(key, hidden.get(key)))
            if isinstance(field, list):
                if (len(field) != 2 or str(field[0]) not in graph or
                    type(field[1]) is not int):
                    raise RuntimeError("Malformed graph link")
                source = graph[str(field[0])]["class_type"]
                outputs = metadata[source]["output"]
                if not 0 <= field[1] < len(outputs):
                    raise RuntimeError(f"Output port not supported by server: {source}")
            elif (isinstance(descriptor, (list, tuple)) and descriptor and
                  isinstance(descriptor[0], (list, tuple))):
                # ComfyUI widget-choice arrays are authoritative for *every*
                # loader, including extension-specific model selectors.
                choices = descriptor[0]
                if field not in choices:
                    raise RuntimeError(
                        f"Unrecognized server widget/model choice for {name}.{key}: {field}"
                    )
    return {
        "checked_classes": len(metadata),
        "node_count": len(graph),
        "server_verified": True,
    }
