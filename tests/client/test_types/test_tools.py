"""Tool.parameters accepts dict, BaseModel class, or TypeAdapter."""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, TypeAdapter

from luca.client.types import Tool, tool_parameters_to_json_schema


class WeatherParams(BaseModel):
    location: str
    units: Literal["celsius", "fahrenheit"] = "celsius"


def test_tool_with_dict_parameters():
    t = Tool(
        name="get_weather",
        description="...",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    schema = tool_parameters_to_json_schema(t.parameters)
    assert schema == {"type": "object", "properties": {"x": {"type": "string"}}}


def test_tool_with_pydantic_basemodel_parameters():
    t = Tool(name="get_weather", description="...", parameters=WeatherParams)
    schema = tool_parameters_to_json_schema(t.parameters)
    assert schema["type"] == "object"
    assert "location" in schema["properties"]


def test_tool_with_typeadapter_parameters():
    Adapter = TypeAdapter(TypedDict("X", {"name": Annotated[str, Field(description="...")]}))
    t = Tool(name="t", description="...", parameters=Adapter)
    schema = tool_parameters_to_json_schema(t.parameters)
    assert "properties" in schema
    assert "name" in schema["properties"]
