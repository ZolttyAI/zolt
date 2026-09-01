"""
Structured, restricted database call parser and schema validator.
Validates <db_call>...</db_call> blocks against a strict schema (dialect, operation, table, constraints).
Restricted strictly to: postgresql | sqlite | mariadb | mysql.
"""
import re
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Set


SUPPORTED_DIALECTS: Set[str] = {"postgresql", "sqlite", "mariadb", "mysql"}
SUPPORTED_OPERATIONS: Set[str] = {"select", "insert", "update", "delete", "create", "alter", "drop"}
REQUIRED_KEYS: Set[str] = {"dialect", "operation", "table", "constraints"}


@dataclass
class DBCallPayload:
    """Validated structured database call payload."""
    dialect: str
    operation: str
    table: str
    constraints: Union[Dict[str, Any], List[Any], str]


def validate_db_call(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a database call payload against the restricted schema.
    Fail-closed validation: rejects missing keys, unsupported dialects, or invalid operations.
    """
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload.strip())
        except json.JSONDecodeError as e:
            return {"valid": False, "payload": None, "error": f"Malformed JSON in db_call: {e}"}
    elif isinstance(payload, dict):
        parsed = payload
    else:
        return {"valid": False, "payload": None, "error": f"Invalid payload type: {type(payload)}"}

    if not isinstance(parsed, dict):
        return {"valid": False, "payload": None, "error": "Database call payload must be a JSON object."}

    # Check required keys
    missing_keys = REQUIRED_KEYS - set(parsed.keys())
    if missing_keys:
        return {"valid": False, "payload": None, "error": f"Missing required keys in db_call: {sorted(list(missing_keys))}"}

    dialect = str(parsed.get("dialect", "")).lower().strip()
    if dialect not in SUPPORTED_DIALECTS:
        return {
            "valid": False,
            "payload": None,
            "error": f"Unsupported SQL dialect '{dialect}'. Supported dialects: {sorted(list(SUPPORTED_DIALECTS))}",
        }

    operation = str(parsed.get("operation", "")).lower().strip()
    if operation not in SUPPORTED_OPERATIONS:
        return {
            "valid": False,
            "payload": None,
            "error": f"Unsupported SQL operation '{operation}'. Supported operations: {sorted(list(SUPPORTED_OPERATIONS))}",
        }

    table = parsed.get("table")
    if not isinstance(table, str) or not table.strip():
        return {"valid": False, "payload": None, "error": "Table name must be a non-empty string."}

    constraints = parsed.get("constraints")
    if constraints is None:
        return {"valid": False, "payload": None, "error": "'constraints' field cannot be None."}

    validated_obj = DBCallPayload(
        dialect=dialect,
        operation=operation,
        table=table.strip(),
        constraints=constraints,
    )

    return {"valid": True, "payload": validated_obj, "error": None}


def parse_db_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract and validate all <db_call> ... </db_call> blocks from text.
    Returns list of validation results for each found block.
    """
    pattern = re.compile(r"<db_call>\n?(.*?)\n?</db_call>", re.DOTALL)
    results = []

    for match in pattern.finditer(text):
        raw_block = match.group(1).strip()
        val_result = validate_db_call(raw_block)
        results.append(val_result)

    return results


def format_db_call(dialect: str, operation: str, table: str, constraints: Any) -> str:
    """Format a database call into standard zolt <db_call> tags after validation."""
    payload = {
        "dialect": dialect.lower().strip(),
        "operation": operation.lower().strip(),
        "table": table.strip(),
        "constraints": constraints,
    }
    val = validate_db_call(payload)
    if not val["valid"]:
        raise ValueError(f"Cannot format invalid db_call: {val['error']}")

    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"<db_call>\n{json_str}\n</db_call>"
