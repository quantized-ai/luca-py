"""The whole real stack, once per provider: `ShellAccessPlugin` with its
REAL tools and middleware, a real runner, a real workspace on disk, and a
scripted native tool call off the wire.

Everything else in this directory swaps the tools for fixtures to keep the
projection assertions readable. This module is the other half: it proves the
shipped composition actually patches a file, stores the native payload, and
re-emits it — the wiring no unit test sees end to end.
"""

from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.shell import ShellAccessPlugin
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.providers.openai import ApplyPatchTool, LocalShellTool
from luca.client.types import ToolCall as ClientToolCall

from .conftest import apply_patch_call, assistant, make_session, plain_call, stored_assistant_parts, text


def shell_runner(tmp_path, model: str) -> PluginAgentSessionRunner:
    """The composition an application writes: one plugin, everything else
    default. YOLO so the battery is about projection, not approvals."""
    session = make_session(model, native=True)
    return PluginAgentSessionRunner(
        session,
        plugins=[ShellAccessPlugin(workspace=tmp_path, mode="yolo")],
    )


async def test_an_openai_native_patch_reaches_disk_and_replays_as_it_arrived(llm, tmp_path):
    (tmp_path / "app.py").write_text("def greet():\n    print('Hi')\n")
    runner = shell_runner(tmp_path, "openai:gpt-5.1")
    llm.queue(
        assistant(
            apply_patch_call(
                "tc1",
                "apc_7",
                {
                    "type": "update_file",
                    "path": "app.py",
                    "diff": "@@ def greet():\n-    print('Hi')\n+    print('Hello')\n",
                },
            ),
            finish_reason="tool_use",
        ),
        assistant(text("done")),
        assistant(text("sure")),
    )
    runner.post_message("say hello instead")
    await runner.run()

    runner.post_message("again")
    await runner.run()

    # 1. the model was offered the native items, not the generics they replace
    assert [tool.name for tool in llm.calls[0]["tools"]] == [
        "read",
        "glob",
        "grep",
        ApplyPatchTool.name,
        LocalShellTool.name,
    ]
    # 2. the call was adopted under its internal name, native payload kept
    assert stored_assistant_parts(runner.session)[0].name == "openai_apply_patch"
    assert stored_assistant_parts(runner.session)[0].extras == {
        "custom_type": "apply_patch_call",
        "item_id": "apc_7",
        "status": "completed",
    }
    # 3. the real tool really patched the file
    assert (tmp_path / "app.py").read_text() == "def greet():\n    print('Hello')\n"
    # 4. and the next request re-emits it exactly as it arrived
    assert llm.calls[-1]["messages"][1].content == [
        ClientToolCall(
            id="tc1",
            name="apply_patch",
            arguments={
                "type": "update_file",
                "path": "app.py",
                "diff": "@@ def greet():\n-    print('Hi')\n+    print('Hello')\n",
            },
            extras={"custom_type": "apply_patch_call", "item_id": "apc_7", "status": "completed"},
        ),
    ]


async def test_an_anthropic_native_edit_reaches_disk_through_the_same_plugin(llm, tmp_path):
    (tmp_path / "app.py").write_text("value = 1\n")
    runner = shell_runner(tmp_path, "anthropic:claude-sonnet-4-5")
    llm.queue(
        assistant(
            plain_call(
                "tc2",
                "str_replace_based_edit_tool",
                {"command": "str_replace", "path": "app.py", "old_str": "value = 1", "new_str": "value = 9"},
            ),
            finish_reason="tool_use",
        ),
        assistant(text("done")),
    )
    runner.post_message("bump it")

    await runner.run()

    assert [tool.name for tool in llm.calls[0]["tools"]] == [
        "read",
        "glob",
        "grep",
        "delete_file",
        TextEditorTool.name,
        BashTool.name,
    ]
    assert stored_assistant_parts(runner.session)[0].name == "anthropic_text_editor_20250728"
    assert (tmp_path / "app.py").read_text() == "value = 9\n"


async def test_the_native_switch_off_runs_the_same_workspace_on_the_generic_tools(llm, tmp_path):
    (tmp_path / "app.py").write_text("value = 1\n")
    session = make_session("openai:gpt-5.1", native=False)
    runner = PluginAgentSessionRunner(
        session,
        plugins=[ShellAccessPlugin(workspace=tmp_path, mode="yolo")],
    )
    llm.queue(
        assistant(
            plain_call("tc3", "edit", {"file_path": str(tmp_path / "app.py"), "old_string": "1", "new_string": "9"}),
            finish_reason="tool_use",
        ),
        assistant(text("done")),
    )
    runner.post_message("bump it")

    await runner.run()

    assert [tool.name for tool in llm.calls[0]["tools"]] == [
        "read",
        "glob",
        "grep",
        "edit",
        "write",
        "apply_patch",
        "delete_file",
        "bash",
    ]
    # the edit tool's read-first guard is untouched by any of this
    assert stored_assistant_parts(runner.session)[0].name == "edit"
    assert (tmp_path / "app.py").read_text() == "value = 1\n"
