"""response_format normalization: the three input styles → (name, schema), and
strict-mode rewriting.

Strict mode is not optional decoration: OpenAI rejects `strict: true` next to a
schema that omits `additionalProperties: false` or leaves a defaulted field out
of `required`, which is exactly what `model_json_schema()` produces."""

import pytest
from pydantic import BaseModel, TypeAdapter

from luca.client.types.structured import (
    response_format_to_json_schema,
    strictify_json_schema,
)


class Movie(BaseModel):
    title: str
    year: int


class Cast(BaseModel):
    lead: str
    extras: list[str] = []


class Film(BaseModel):
    movie: Movie
    cast: Cast


def test_a_pydantic_model_lends_its_class_name_to_the_schema():
    assert response_format_to_json_schema(Movie) == (
        "Movie",
        {
            "properties": {
                "title": {"title": "Title", "type": "string"},
                "year": {"title": "Year", "type": "integer"},
            },
            "required": ["title", "year"],
            "title": "Movie",
            "type": "object",
        },
    )


def test_a_raw_dict_is_the_schema_itself():
    # A dict is ALWAYS a JSON Schema — never a wire-level response_format
    # payload. Wire overrides go through provider_options.
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}

    assert response_format_to_json_schema(schema) == ("structured_output", schema)


def test_a_type_adapter_contributes_its_json_schema():
    name, schema = response_format_to_json_schema(TypeAdapter(list[int]))

    assert (name, schema) == ("structured_output", {"items": {"type": "integer"}, "type": "array"})


def test_an_unsupported_form_is_refused():
    with pytest.raises(TypeError, match="must be a dict, BaseModel subclass, or TypeAdapter"):
        response_format_to_json_schema("Movie")


def test_strictify_requires_every_property_and_forbids_extras():
    assert strictify_json_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
    ) == {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }


def test_strictify_reaches_defs_and_nested_objects():
    # `$ref` is left alone and the referenced `$defs` entry is rewritten in
    # place, which is what the reference resolves to. `Cast.extras` has a
    # default, so model_json_schema() leaves it out of `required` and strict
    # mode puts it back.
    _, schema = response_format_to_json_schema(Film)

    assert strictify_json_schema(schema) == {
        "$defs": {
            "Cast": {
                "properties": {
                    "lead": {"title": "Lead", "type": "string"},
                    "extras": {
                        "default": [],
                        "items": {"type": "string"},
                        "title": "Extras",
                        "type": "array",
                    },
                },
                "required": ["lead", "extras"],
                "title": "Cast",
                "type": "object",
                "additionalProperties": False,
            },
            "Movie": {
                "properties": {
                    "title": {"title": "Title", "type": "string"},
                    "year": {"title": "Year", "type": "integer"},
                },
                "required": ["title", "year"],
                "title": "Movie",
                "type": "object",
                "additionalProperties": False,
            },
        },
        "properties": {"movie": {"$ref": "#/$defs/Movie"}, "cast": {"$ref": "#/$defs/Cast"}},
        "required": ["movie", "cast"],
        "title": "Film",
        "type": "object",
        "additionalProperties": False,
    }


def test_strictify_reaches_objects_inside_arrays_and_unions():
    # The `null` branch of the union is not an object and is left untouched.
    assert strictify_json_schema(
        {
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": {"type": "object", "properties": {"a": {"type": "string"}}}},
                "either": {
                    "anyOf": [
                        {"type": "object", "properties": {"b": {"type": "string"}}},
                        {"type": "null"},
                    ]
                },
            },
        }
    ) == {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                    "additionalProperties": False,
                },
            },
            "either": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"b": {"type": "string"}},
                        "required": ["b"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
        },
        "required": ["rows", "either"],
        "additionalProperties": False,
    }


def test_strictify_is_idempotent():
    _, schema = response_format_to_json_schema(Film)

    once = strictify_json_schema(schema)

    assert strictify_json_schema(once) == once


def test_strictify_does_not_mutate_its_input():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}

    strictify_json_schema(schema)

    assert schema == {"type": "object", "properties": {"a": {"type": "string"}}}
