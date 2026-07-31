"""response_format normalization: the three input styles → (name, schema), and
strict-mode rewriting.

Strict mode is not optional decoration: OpenAI rejects `strict: true` next to a
schema that omits `additionalProperties: false` or leaves a defaulted field out
of `required`, which is exactly what `model_json_schema()` produces."""

import pytest
from pydantic import BaseModel, Field, TypeAdapter

from luca.client.types.structured import (
    response_format_to_json_schema,
    sanitize_schema_name,
    strictify_json_schema,
    strip_unsupported_keywords,
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


class Described(BaseModel):
    inner: Movie = Field(description="a described submodel")


class Node(BaseModel):
    value: int
    child: "Node | None" = Field(default=None, description="the child node")


Node.model_rebuild()


def test_a_ref_with_sibling_keys_is_inlined():
    # `Field(description=…)` on a nested model produces {"$ref", "description"},
    # which OpenAI rejects outright: "$ref cannot have keywords".
    _, schema = response_format_to_json_schema(Described)

    inner = strictify_json_schema(schema)["properties"]["inner"]

    assert inner == {
        "type": "object",
        "title": "Movie",
        "properties": {
            "title": {"title": "Title", "type": "string"},
            "year": {"title": "Year", "type": "integer"},
        },
        "required": ["title", "year"],
        "additionalProperties": False,
        "description": "a described submodel",
    }


def test_a_bare_ref_is_left_alone():
    _, schema = response_format_to_json_schema(Film)

    assert strictify_json_schema(schema)["properties"]["movie"] == {"$ref": "#/$defs/Movie"}


def test_a_recursive_model_terminates_and_stays_idempotent():
    # Inlining goes one level only, so the self-reference inside the inlined
    # target stays a ref instead of expanding forever.
    _, schema = response_format_to_json_schema(Node)

    strict = strictify_json_schema(schema)

    assert strictify_json_schema(strict) == strict
    assert strict["$defs"]["Node"]["additionalProperties"] is False


def test_an_object_with_no_properties_is_still_rewritten():
    # It used to pass through untouched while `strict: true` went out beside
    # it, which is the exact 400 the rewrite exists to prevent.
    assert strictify_json_schema({"type": "object"}) == {
        "type": "object",
        "required": [],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Box[int]", "Box_int_"),
        ("Café", "Caf_"),
        ("A" * 70, "A" * 64),
        ("!!!", "___"),
        ("", "structured_output"),
    ],
    ids=["generic", "unicode", "too_long", "all_illegal", "empty"],
)
def test_the_schema_name_is_reduced_to_what_the_wire_accepts(raw, expected):
    assert sanitize_schema_name(raw) == expected


def test_anthropic_unsupported_constraints_move_into_the_description():
    # Anthropic's grammar 400s on these; their own SDKs strip and describe.
    assert strip_unsupported_keywords(
        {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
                "name": {"type": "string", "minLength": 2, "description": "the name"},
            },
        }
    ) == {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "description": "Constraints: minimum: 0, maximum: 120"},
            "name": {"type": "string", "description": "the name (minLength: 2)"},
        },
    }


def test_the_keywords_anthropic_does_support_survive():
    schema = {
        "type": "object",
        "properties": {
            "when": {"type": "string", "format": "date-time"},
            "tags": {"type": "array", "minItems": 1},
            "kind": {"type": "string", "enum": ["a", "b"]},
        },
    }

    assert strip_unsupported_keywords(schema) == schema


def test_an_unsupported_format_and_an_out_of_range_min_items_are_stripped():
    assert strip_unsupported_keywords(
        {"type": "object", "properties": {"p": {"type": "string", "format": "binary"}, "t": {"minItems": 3}}}
    ) == {
        "type": "object",
        "properties": {
            "p": {"type": "string", "description": "Constraints: format: binary"},
            "t": {"description": "Constraints: minItems: 3"},
        },
    }
