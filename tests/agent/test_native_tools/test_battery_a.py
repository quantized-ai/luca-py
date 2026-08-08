"""Battery A — advertisement (UC0).

Given an empty conversation and one active config, the FULL `tools` list
`acompletion()` receives: the plugin's `build_tool_list` drops (spec
vocabulary), the new `adapt_tool_declarations` slot swaps the surviving
native declarations for the client's native items (client vocabulary), and
everything else rides through as an ordinary function declaration.
"""

from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.providers.openai import ApplyPatchTool, LocalShellTool

from .conftest import assistant, make_runner, make_session, text, wire_tool
from .tools import ApplyPatchGenericTool, BashGenericTool, DeleteFileTool, ReadTool

GENERIC_WIRE_TOOLS = [
    wire_tool(ReadTool),
    wire_tool(DeleteFileTool),
    wire_tool(ApplyPatchGenericTool),
    wire_tool(BashGenericTool),
]


async def test_a1_openai_native_advertises_the_native_items_plus_read(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    llm.queue(assistant(text("hi")))
    runner.post_message("hello")

    await runner.run()

    assert llm.calls[0]["model"] == "openai:gpt-5.1"
    assert llm.calls[0]["tools"] == [ApplyPatchTool(), LocalShellTool(), wire_tool(ReadTool)]


async def test_a2_anthropic_native_advertises_the_native_items_plus_read_and_delete_file(llm):
    session = make_session("anthropic:claude-sonnet-4-5", native=True)
    runner = make_runner(session)
    llm.queue(assistant(text("hi")))
    runner.post_message("hello")

    await runner.run()

    # `text_editor view` reads text but returns no images and feeds no read
    # tracker, and it cannot delete — so both generics survive.
    assert llm.calls[0]["model"] == "anthropic:claude-sonnet-4-5"
    assert llm.calls[0]["tools"] == [
        TextEditorTool(),
        BashTool(),
        wire_tool(ReadTool),
        wire_tool(DeleteFileTool),
    ]


async def test_a3_a_provider_with_no_natives_gets_every_generic(llm):
    session = make_session("openrouter:kimi-2.7", native=True)
    runner = make_runner(session)
    llm.queue(assistant(text("hi")))
    runner.post_message("hello")

    await runner.run()

    assert llm.calls[0]["tools"] == GENERIC_WIRE_TOOLS


async def test_a4_native_off_gets_every_generic_even_on_openai(llm):
    session = make_session("openai:gpt-5.1", native=False)
    runner = make_runner(session)
    llm.queue(assistant(text("hi")))
    runner.post_message("hello")

    await runner.run()

    assert llm.calls[0]["tools"] == GENERIC_WIRE_TOOLS


async def test_a5_flipping_native_off_mid_session_takes_effect_on_the_next_run(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    llm.queue(assistant(text("hi")), assistant(text("hi again")))
    runner.post_message("hello")
    await runner.run()

    session.session_config.use_native_tools = False
    runner.post_message("and now?")
    await runner.run()

    # the per-iteration re-stamp: same session, same runner, generics only
    assert llm.calls[0]["tools"] == [ApplyPatchTool(), LocalShellTool(), wire_tool(ReadTool)]
    assert llm.calls[1]["tools"] == GENERIC_WIRE_TOOLS
