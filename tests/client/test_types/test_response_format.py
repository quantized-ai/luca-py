"""response_format parsing via response.parse()."""

import pytest
from pydantic import BaseModel

from luca.client.exceptions import StructuredOutputError
from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    TextBlock,
)


class Movie(BaseModel):
    title: str
    year: int


def test_parse_returns_pydantic_instance():
    msg = AssistantMessage(content=[TextBlock(text='{"title":"Hi","year":1999}')])
    response = ChatCompletionResponse(message=msg)
    response._response_format = Movie
    movie = response.parse()
    assert isinstance(movie, Movie)
    assert movie.title == "Hi"
    assert movie.year == 1999


def test_parse_with_dict_schema_returns_dict():
    msg = AssistantMessage(content=[TextBlock(text='{"a":1}')])
    response = ChatCompletionResponse(message=msg)
    response._response_format = {"type": "object"}
    result = response.parse()
    assert result == {"a": 1}


def test_parse_without_response_format_raises_valueerror():
    msg = AssistantMessage(content=[TextBlock(text='{"a":1}')])
    response = ChatCompletionResponse(message=msg)
    with pytest.raises(ValueError, match="response_format"):
        response.parse()


def test_parse_with_invalid_json_raises_structured_output_error():
    msg = AssistantMessage(content=[TextBlock(text="not json")])
    response = ChatCompletionResponse(message=msg)
    response._response_format = Movie
    with pytest.raises(StructuredOutputError):
        response.parse()


def test_parse_with_invalid_data_raises_structured_output_error():
    msg = AssistantMessage(content=[TextBlock(text='{"title":"Hi"}')])
    response = ChatCompletionResponse(message=msg)
    response._response_format = Movie
    with pytest.raises(StructuredOutputError):
        response.parse()
