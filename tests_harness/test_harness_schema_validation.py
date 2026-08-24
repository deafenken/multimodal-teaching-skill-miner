from __future__ import annotations

import unittest

from agent_harness.core import HarnessContractError, ToolSpec
from agent_harness.core.schema import validate_schema


class HarnessToolSchemaValidationTests(unittest.TestCase):
    def test_tool_registration_rejects_unknown_nested_keywords(self) -> None:
        with self.assertRaisesRegex(HarnessContractError, "unsupported schema"):
            ToolSpec(
                name="invalid_nested",
                version="1.0.0",
                description="Invalid schema fixture.",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string", "format": "email"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            )

    def test_tool_registration_rejects_invalid_schema_constraints(self) -> None:
        invalid_schemas = (
            {"type": "string", "pattern": "["},
            {"type": "string", "minLength": 4, "maxLength": 2},
            {"type": "array", "uniqueItems": "yes"},
            {"type": [{}]},
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["missing"],
            },
        )
        for index, schema in enumerate(invalid_schemas):
            with self.subTest(index=index):
                with self.assertRaises(HarnessContractError):
                    ToolSpec(
                        name=f"invalid_{index}",
                        version="1.0.0",
                        description="Invalid schema fixture.",
                        input_schema=schema,
                    )

    def test_unique_items_uses_canonical_json_not_mapping_repr(self) -> None:
        schema = {
            "type": "array",
            "items": {"type": "object"},
            "uniqueItems": True,
        }
        with self.assertRaisesRegex(HarnessContractError, "must be unique"):
            validate_schema([{"a": 1, "b": 2}, {"b": 2, "a": 1}], schema)

    def test_valid_nested_tool_contract_still_registers(self) -> None:
        specification = ToolSpec(
            name="valid_nested",
            version="1.0.0",
            description="Valid schema fixture.",
            input_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
        )
        self.assertEqual(specification.name, "valid_nested")


if __name__ == "__main__":
    unittest.main()
