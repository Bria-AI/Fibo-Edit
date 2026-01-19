from typing import Any

from pydantic import BaseModel


class SchemaBuilder(BaseModel):
    @staticmethod  # Change to staticmethod
    def build_instruction_schema(target_cls: Any) -> str: # Accept the target class
        schema_lines = []
        # Access the fields of the class passed in
        for i, (name, field) in enumerate(target_cls.model_fields.items(), 1):
            description = field.description or "No description provided."
            schema_lines.append(f"{i}. `{name}`: {description}")
        return "\n".join(schema_lines)