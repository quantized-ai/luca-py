"""Tool.parameters accepts dict, BaseModel class, or TypeAdapter; plus the
shared ApproximateLocation value object."""

from typing import Annotated, Literal, TypedDict

import pytest
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from luca.client.types import ApproximateLocation, Tool, tool_parameters_to_json_schema


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


def test_approximate_location_needs_at_least_one_field():
    with pytest.raises(ValidationError, match="at least one"):
        ApproximateLocation()


def test_approximate_location_with_one_field_is_valid():
    assert ApproximateLocation(country="US").country == "US"
