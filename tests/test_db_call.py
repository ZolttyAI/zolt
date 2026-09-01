"""Unit tests for structured restricted database calls."""
import pytest
from zolt.inference.db_call import (
    validate_db_call,
    parse_db_calls,
    format_db_call,
    SUPPORTED_DIALECTS,
)
from zolt.data.db_call_synth import generate_db_call_examples


def test_validate_db_call_supported_dialects():
    for dialect in ["postgresql", "sqlite", "mariadb", "mysql"]:
        payload = {
            "dialect": dialect,
            "operation": "select",
            "table": "users",
            "constraints": {"id": 1},
        }
        res = validate_db_call(payload)
        assert res["valid"], f"Failed for supported dialect {dialect}: {res.get('error')}"
        assert res["payload"].dialect == dialect


def test_validate_db_call_unsupported_dialect():
    payload = {
        "dialect": "oracle",
        "operation": "select",
        "table": "users",
        "constraints": {},
    }
    res = validate_db_call(payload)
    assert not res["valid"]
    assert "Unsupported SQL dialect" in res["error"]


def test_validate_db_call_unsupported_operation():
    payload = {
        "dialect": "postgresql",
        "operation": "truncate_cascade_hack",
        "table": "users",
        "constraints": {},
    }
    res = validate_db_call(payload)
    assert not res["valid"]
    assert "Unsupported SQL operation" in res["error"]


def test_validate_db_call_missing_required_keys():
    payload = {
        "dialect": "sqlite",
        "operation": "select",
        # missing table and constraints
    }
    res = validate_db_call(payload)
    assert not res["valid"]
    assert "Missing required keys" in res["error"]


def test_validate_db_call_malformed_json_string():
    res = validate_db_call("{dialect: postgresql, missing_quotes}")
    assert not res["valid"]
    assert "Malformed JSON" in res["error"]


def test_parse_db_calls_block():
    text = (
        "Here is the database execution plan:\n"
        "<db_call>\n"
        "{\n"
        '  "dialect": "postgresql",\n'
        '  "operation": "insert",\n'
        '  "table": "accounts",\n'
        '  "constraints": {"returning": ["id"]}\n'
        "}\n"
        "</db_call>\n"
        "Done."
    )
    parsed = parse_db_calls(text)
    assert len(parsed) == 1
    assert parsed[0]["valid"]
    assert parsed[0]["payload"].table == "accounts"


def test_synthetic_db_call_generator():
    examples = generate_db_call_examples(n_copies=2)
    assert len(examples) > 0
    for ex in examples:
        assert ex["dialect"] in SUPPORTED_DIALECTS
        assert "<db_call>" in ex["content"]
        assert "</db_call>" in ex["content"]
        assert "<think>" in ex["content"]
