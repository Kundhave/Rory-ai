"""Tool registration, schema generation, argument validation, and dispatch.

The security boundary lives here. Validation runs to completion *before* a tool
body executes, so a value the model invented can never reach the code that would
act on it. Enum membership is checked against the schema, and the schema's enum
is generated from the whitelist itself — there is no second list to drift.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

# Tool output is untrusted, unbounded text as far as the prompt is concerned.
MAX_RESULT_CHARS = 2000

_JSON_TYPES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., dict]


_REGISTRY: dict[str, Tool] = {}


def tool(fn: Callable[..., dict]) -> Callable[..., dict]:
    _REGISTRY[fn.__name__] = Tool(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        schema=_build_schema(fn),
        fn=fn,
    )
    return fn


def schemas() -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "parameters": t.schema}
        for t in _REGISTRY.values()
    ]


def dispatch(name: str, arguments: dict) -> dict:
    """Never raises. A broken tool becomes a result the model can explain."""
    target = _REGISTRY.get(name)
    if target is None:
        return {"ok": False, "error": f"unknown tool: {name!r}"}

    rejection = _validate(target.schema, arguments)
    if rejection:
        return {"ok": False, "error": rejection}

    try:
        return target.fn(**arguments)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def truncate_result(result: dict) -> str:
    payload = json.dumps(result)
    if len(payload) <= MAX_RESULT_CHARS:
        return payload
    return payload[:MAX_RESULT_CHARS] + "...[truncated]"


def _build_schema(fn: Callable[..., dict]) -> dict:
    hints = get_type_hints(fn)
    names = list(inspect.signature(fn).parameters)
    return {
        "type": "object",
        "properties": {name: _param_schema(hints[name]) for name in names},
        "required": names,
    }


def _param_schema(hint: Any) -> dict:
    if get_origin(hint) is Literal:
        return {"type": "string", "enum": list(get_args(hint))}
    return {"type": _JSON_TYPES[hint]}


def _validate(schema: dict, arguments: dict) -> str | None:
    properties: dict[str, dict] = schema["properties"]

    unexpected = sorted(set(arguments) - set(properties))
    if unexpected:
        return f"unexpected argument(s): {unexpected}"

    missing = [name for name in schema["required"] if name not in arguments]
    if missing:
        return f"missing argument(s): {missing}"

    for name, spec in properties.items():
        value = arguments[name]
        if "enum" in spec and value not in spec["enum"]:
            return f"{name}={value!r} is not one of {spec['enum']}"
        if not _type_matches(spec["type"], value):
            return f"{name} must be a {spec['type']}, got {type(value).__name__}"
    return None


def _type_matches(json_type: str, value: Any) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, bool)
