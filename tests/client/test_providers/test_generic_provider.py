"""GenericProvider — per-instance attrs."""

from luca.client.providers import GenericProvider
from luca.client.transports import OpenAITransport


def test_generic_provider_uses_per_instance_attrs():
    g = GenericProvider(
        name="groq", default_base_url="https://api.groq.com/openai/v1",
        default_api_key_env_var="GROQ_API_KEY",
        default_transport_class=OpenAITransport, api_key="grq",
    )
    assert g.name == "groq"
    assert g.transport._provider == "groq"
    assert g.transport._base_url == "https://api.groq.com/openai/v1"
    assert isinstance(g.transport, OpenAITransport)


def test_generic_provider_instances_do_not_pollute_each_other():
    a = GenericProvider(
        name="a", default_base_url="https://a.test",
        default_api_key_env_var=None,
        default_transport_class=OpenAITransport, api_key="ka",
    )
    b = GenericProvider(
        name="b", default_base_url="https://b.test",
        default_api_key_env_var=None,
        default_transport_class=OpenAITransport, api_key="kb",
    )
    assert (a.name, a.transport._base_url) == ("a", "https://a.test")
    assert (b.name, b.transport._base_url) == ("b", "https://b.test")
