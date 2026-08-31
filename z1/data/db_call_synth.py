"""
Synthetic dataset generator for structured database calls (<db_call>).
Produces examples contrasting SQL dialect nuances strictly across PostgreSQL, SQLite, MariaDB, and MySQL.
Explicitly out of scope: NoSQL, schema-design optimization.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from z1.inference.db_call import format_db_call, SUPPORTED_DIALECTS


DIALECT_EXAMPLES = [
    {
        "dialect": "postgresql",
        "operation": "insert",
        "table": "users",
        "constraints": {"on_conflict": "DO UPDATE SET updated_at = NOW()", "returning": ["id", "created_at"]},
        "reasoning": "PostgreSQL supports ON CONFLICT (upsert) and RETURNING clauses for atomic insert-and-fetch operations.",
        "sql": "INSERT INTO users (email, name) VALUES ('user@example.com', 'Dev') ON CONFLICT (email) DO UPDATE SET updated_at = NOW() RETURNING id, created_at;",
    },
    {
        "dialect": "sqlite",
        "operation": "insert",
        "table": "users",
        "constraints": {"or_action": "OR IGNORE", "primary_key": "id AUTOINCREMENT"},
        "reasoning": "SQLite supports INSERT OR IGNORE and AUTOINCREMENT on INTEGER PRIMARY KEY columns.",
        "sql": "INSERT OR IGNORE INTO users (email, name) VALUES ('user@example.com', 'Dev');",
    },
    {
        "dialect": "mysql",
        "operation": "insert",
        "table": "users",
        "constraints": {"on_duplicate": "UPDATE name = VALUES(name), updated_at = CURRENT_TIMESTAMP"},
        "reasoning": "MySQL uses ON DUPLICATE KEY UPDATE syntax rather than SQL-standard ON CONFLICT.",
        "sql": "INSERT INTO users (email, name) VALUES ('user@example.com', 'Dev') ON DUPLICATE KEY UPDATE name = VALUES(name), updated_at = CURRENT_TIMESTAMP;",
    },
    {
        "dialect": "mariadb",
        "operation": "select",
        "table": "audit_logs",
        "constraints": {"limit": 100, "offset": 0, "window_functions": True},
        "reasoning": "MariaDB supports window functions and RETURNING with INSERT/DELETE/UPDATE statements.",
        "sql": "SELECT id, user_id, action, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC) as rn FROM audit_logs LIMIT 100;",
    },
    {
        "dialect": "postgresql",
        "operation": "select",
        "table": "events",
        "constraints": {"json_filter": "payload->>'status' = 'active'", "limit": 50},
        "reasoning": "PostgreSQL provides native JSONB operator ->> for text extraction within indexed queries.",
        "sql": "SELECT id, payload FROM events WHERE payload->>'status' = 'active' LIMIT 50;",
    },
    {
        "dialect": "sqlite",
        "operation": "create",
        "table": "settings",
        "constraints": {"columns": {"key": "TEXT PRIMARY KEY", "value": "TEXT NOT NULL", "updated_at": "DATETIME DEFAULT CURRENT_TIMESTAMP"}},
        "reasoning": "SQLite uses dynamic type affinity with standard type declarations.",
        "sql": "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP);",
    },
    {
        "dialect": "mysql",
        "operation": "update",
        "table": "orders",
        "constraints": {"set": {"status": "'shipped'"}, "where": {"order_id": 1024, "status": "'pending'"}},
        "reasoning": "MySQL UPDATE syntax with indexed WHERE conditions for consistent isolation.",
        "sql": "UPDATE orders SET status = 'shipped' WHERE order_id = 1024 AND status = 'pending';",
    },
    {
        "dialect": "mariadb",
        "operation": "delete",
        "table": "expired_tokens",
        "constraints": {"where": "expires_at < NOW()", "returning": ["token_id"]},
        "reasoning": "MariaDB supports RETURNING on DELETE statements to identify purged records.",
        "sql": "DELETE FROM expired_tokens WHERE expires_at < NOW() RETURNING token_id;",
    },
]


def generate_db_call_examples(n_copies: int = 10) -> List[Dict[str, Any]]:
    """Generate synthetic training records demonstrating structured DB calls."""
    records = []
    for _ in range(n_copies):
        for ex in DIALECT_EXAMPLES:
            db_block = format_db_call(
                dialect=ex["dialect"],
                operation=ex["operation"],
                table=ex["table"],
                constraints=ex["constraints"],
            )

            prompt = f"Execute a {ex['operation']} operation on the '{ex['table']}' table using {ex['dialect'].capitalize()}."
            content = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"<think>\n{ex['reasoning']}\n</think>\n"
                f"{db_block}\n"
                f"<code>\n{ex['sql']}\n</code><|im_end|>"
            )

            records.append({
                "content": content,
                "lang": "sql",
                "license": "synthetic",
                "quality_score": 1.0,
                "dialect": ex["dialect"],
            })

    return records


def save_synthetic_db_dataset(output_path: str, n_copies: int = 10) -> int:
    """Save synthetic DB call training pairs to JSONL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = generate_db_call_examples(n_copies=n_copies)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[z1-db-synth] Generated {len(records)} SQL dialect synthetic instances -> {output_path}")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="z1 SQL Dialect Synthetic Data Generator")
    parser.add_argument("--output", default="data/distilled/db_calls_synthetic.jsonl", help="Output JSONL path")
    parser.add_argument("--copies", type=int, default=10, help="Number of repetitions per example")
    args = parser.parse_args()

    save_synthetic_db_dataset(args.output, n_copies=args.copies)
