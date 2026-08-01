I want to implement a new major feature: subagents. Which is a basic feature of all AI Coding agents. The LLM can choose to spawn parallel subagents that take care of tasks in parallel.

Let's brainstorm the architectural decisions for our implementation of subagents. I have everything mostly figured out, but I need you to review it and validate that the data model and architecture is correct. I want to keep this first version AS SIMPLE as possible and focus only on the Data Model, Spec, behavior and architecture.

## Data model
The easiest thing given how robust our data model is, is to add a new Entry type which is something like `ChildConversation`. Then move around a bit the attributes of AgentSession but end up with something like this:

```python
class ChildConversation(Entry):
    type: Literal["child_conversation"] = "child_conversation"
    conversation_id: str
    tool_execution_id: str # track which tool created this child convo
    execution_result: ExecutionResult | None = None
    # maybe some other param, like timeout?

class AgentSession(BaseModel):
    active_conversation_id: str
    conversation_history: List[str] # list of conversation ids
    conversations: list[Conversation] = Field(default_factory=list) # the "store/catalog of ALL conversations"
```

So then everything can be reused: our data model is already built in a robust ways with "stores" on one side and "nodes" that make sense of that store on the other. So there's no need to spawn a different session duplicating for example ToolSpec if they're already cached in the main session.

The LLM replying with parallel tool calls closes the ToolExecution immediately, because conceptually the responsibility of the tool call is to CREATE the conversation, so it is indeed "completed". That's the scope of the ToolExecution. But the turn is kept "open" and the "main conversation" is still "RUNNING"

Here's the "timeline" of how this would unfold. The user prompts and the LLM replies with the parallel subagents:

```
main_conversation = #C1
conversations:
    #C1: u1 > t1 > A1(tc1, tc2) | Status: Running

Nodes:
u1: UserMessage("Research A and B in parallel, then compare")
t1: TurnStart()
A1:
    tc1: spawn_agent(A) - ToolSpec: #spec1
    tc2: spawn_agent(B) - ToolSpec: #spec1
    finish_reason: tool_call

ToolSpecs:
#spec1: spawn_agents(prompt: str, ...)

```

So far the turn is open, the runner now creates and executes the functions, which "complete" pretty much immediately (if `max_subagents` is not exceeded, and some other checks, not important now). The tool calls will create the ChildConversations in status "Running" The session status will be now:

```
main_conversation = #C1
conversations:
    #C1: u1 > t1 > A1(tc1, tc2) > TE1 > TE2 > CC2 > CC3 | Status: Running
    #C2: u2 > t2 | Status: Running
    #C3: u3 > t3 | Status: Running

Nodes:
u1: UserMessage("Research A and B in parallel, then compare")
t1: TurnStart()
A1:
    tc1: spawn_agent(A) - ToolSpec: #spec1
    tc2: spawn_agent(B) - ToolSpec: #spec1
    finish_reason: tool_call

TE1: ToolExecution(spawn_agent(A), status=COMPLETED)
TE2: ToolExecution(spawn_agent(A), status=COMPLETED)
CC2: conversation_id: #C2
CC3: conversation_id: #C3

u2: "You are a subagent, you're tasked with researching A"
u3: "You are a subagent, you're tasked with researching B"
t2: TurnStart()
t3: TurnStart()

ToolSpecs:
#spec1: spawn_agents(prompt: str, ...)

```

The important part is that in this case all the conversations are in running. The tool execution is COMPLETED, because indeed the tool scope is done.
The runner has to keep now executing everything in parallel based on the logic we're expecting. The user could decide to interrupt 1 subagent or all. The runner could also assign timeouts and all that gist.

The next step in this progression is the runner keeps the "main" conversation #C1 on hold "blocked?" while it executes the children conversations, so let's say they return some other tool calls:

```
main_conversation = #C1
conversations:
    #C1: u1 > t1 > A1(tc1, tc2) > TE1 > TE2 > CC2 > CC3 | Status: Running
    #C2: u2 > t2 > A2(tc3, tc4) > TE3 > TE4 | Status: Running
    #C3: u3 > t3 > A3(tc5) > TE5 | Status: Running

Nodes:
u1: UserMessage("Research A and B in parallel, then compare")
t1: TurnStart()
A1:
    tc1: spawn_agent(A) - ToolSpec: #spec1
    tc2: spawn_agent(B) - ToolSpec: #spec1
    finish_reason: tool_call

TE1: ToolExecution(spawn_agent(A), status=COMPLETED)
TE2: ToolExecution(spawn_agent(A), status=COMPLETED)
CC2: conversation_id: #C2
CC3: conversation_id: #C3

u2: "You are a subagent, you're tasked with researching A"
u3: "You are a subagent, you're tasked with researching B"
t2: TurnStart()
t3: TurnStart()

A2:
    tc3: read_file(main.py) - ToolSpec #read_file
    tc4: read_file(pyproject.toml) - ToolSpec #read_file

A3:
    tc5: read_file(uv.lock) - ToolSpec #read_file

TE3: ToolExecution(read_file(main.py), status=PENDING)
TE4: ToolExecution(read_file(pyproject.toml), status=PENDING)
TE5: ToolExecution(read_file(uv.lock), status=PENDING)

ToolSpecs:
#spec1: spawn_agents(prompt: str, ...)
#read_file: read_file(path: str, ...)

```

At some point the children conversations "finish", the hint that the subagent "completed" is that the children conversation has a TurnFinish, so they're in "IDLE" status, as a child conversation CAN'T get "user messages" posted (only the full turn of the LLM with tool calls and all that). This would be the final status:

```
main_conversation = #C1
conversations:
    #C1: u1 > t1 > A1(tc1, tc2) > TE1 > TE2 > CC2 > CC3 > A4 | Status: Running
    #C2: u2 > t2 > A2(tc3, tc4) > TE3 > TE4 > AC2 > TF_C2| Status: Idle
    #C3: u3 > t3 > A3(tc5) > TE5 > AC3 > TF_C3 | Status: Idle

Nodes:
u1: UserMessage("Research A and B in parallel, then compare")
t1: TurnStart()
A1:
    tc1: spawn_agent(A) - ToolSpec: #spec1
    tc2: spawn_agent(B) - ToolSpec: #spec1
    finish_reason: tool_call

TE1: ToolExecution(spawn_agent(A), status=COMPLETED)
TE2: ToolExecution(spawn_agent(A), status=COMPLETED)
CC2: conversation_id: #C2
CC3: conversation_id: #C3

u2: "You are a subagent, you're tasked with researching A"
u3: "You are a subagent, you're tasked with researching B"
t2: TurnStart()
t3: TurnStart()

A2:
    tc3: read_file(main.py) - ToolSpec #read_file
    tc4: read_file(pyproject.toml) - ToolSpec #read_file

A3:
    tc5: read_file(uv.lock) - ToolSpec #read_file

TE3: ToolExecution(read_file(main.py), status=COMPLETED, output=...)
TE4: ToolExecution(read_file(pyproject.toml), status=COMPLETED, output=...)
TE5: ToolExecution(read_file(uv.lock), status=COMPLETED, output=...)

AC2: # assistant message wrapping up the subagent
    Text: "I found the issue reading main.py and pyproject.toml. The issue is ..."

AC3: # assistant message wrapping up the subagent
    Text: "I read uv.lock but couldn't find anything"

TF_C2: TurnFinish(outcome=COMPLETED)
TF_C3: TurnFinish(outcome=COMPLETED)

A4:
    Reasoning: The subagents found something interesting... i better investigate more
ToolSpecs:
#spec1: spawn_agents(prompt: str, ...)
#read_file: read_file(path: str, ...)

```

The key is that the projector can now project the whole child conversation as desired, and can be configured. So for the "main turn" the projector can project the ToolCalls that originated the subagents as simple outputs, and the whole conversations as user messages with the result. Here's the projected (one example of many options) of the main turn at this point:

```
User: Research A and B in parallel, then compare
Assistant:
    tool_call_1: spawn_agent("Research A")
    tool_call_2: spawn_agent("Research B")
User:
    tool_call_1: content: "COMPLETED" # here we can infer from conversation.STATUS if it was cancelled or interrupted or failed or whatever
    tool_call_2: content: "COMPLETED"
User:
    content: """The result of the subagents was:
<task id=tool_call_id>
    <tool_call tc3 read_file(main.py) contents=...>
    <tool_call tc4 read_file(pyproject.toml) contents=...>
    <text>I found the issue reading main.py and pyproject.toml. The issue is ...</text>
</task>
<task id=tool_call_id>
    <tool_call tc3 read_file(uv.lock) contents=...>
    <text>I read uv.lock but couldn't find anything</text>
</task>
"""
```

### Context management Compaction

The ContextManager has to take care of the compaction mechanisms and calculate `context_tokens`. For compaction, for now, it can use the Result if any, or just ignore the subagent conversations entirely. The point is that this can be improved later. We have to take care of the data model and architecture and ensure that the V0 implementation supports extending and improving without changing those foundational pieces.

### Usage
We have to make sure Usage is properly calculated and reported back to the client adding/summing up the usage tokens of all the main conversation + the nested conversations. The data model supports this automatically, we shouldn't need to change anything, just test it explicitly to make sure nothing breaks. We have to mock a deterministic AgentSession up to spawn subagents and then the Fauxclient with the subagent results + usages and ensure they're stored correctly and the total is accurate.


## Runtime configuration

The `RuntimeConfig` will have to have a few new properties all prefixed with `subagents_`:

- `subagents_enabled: bool = default(False)`
- `subagents_max_depth: int = 1`: for now it's the ONLY allowed value. The data model supports infinite nested subagents, but for now the first implementation will support only `subagents_max_depth=1`. This is an important decision, the runner can be implemented considering this.
- `subagent_soft_max_steps` and `subagent_hard_max_steps`. Equivalent to current values in RuntimeConfig. If not present they take the default values of the main runtime conf.

## Tool implementation

My current best solution is to decouple the Tool from the actual runner execution. So there will be a new plugin `subagents` that will be defined with:

- `get_system_prompt_parts`: returns some more info for the system prompt so LLMs are aware of the subagents feature
- The tool registry, that for now will contain 2 tools: SpawnSubagent and CreateSubagentResult

```python
class SpawnSubagent(Tool):
    namespace = "contrib.subagents"
    name = "spawn_subagent"
    description = "Spawn a subagent"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        task_id: str | None = Field(description="Make up a unique id for the task or we'll make it up for you", default=None)
        prompt: str = Field(description="The prompt for the subagent")
        description: str = Field(description="The description of the task")

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        task_id = args.get('task_id', random_id())
        prompt = args['prompt']
        description = args['description']
        return ExecutionResult(
            content=[TextContent(text=f"Spawned subagent: {description}")],
            metadata={
                "task_id": task_id,
                "prompt": prompt,
                "description": description,
                "process_subagent_result_tool_name": "create_conversation_result"
            }
        )

class SpawnSubagent(Tool):
    namespace = "contrib.subagents"
    name = "create_conversation_result"
    description = "Derive the result of the subagent. LLMs should NOT invoke this tool"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        task_id: str = Field(description="Make up a unique id for the task or we'll make it up for you")
        prompt: str = Field(description="The prompt for the subagent")
        description: str = Field(description="The description of the task")
        conversation_id: str

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        task_id = args.get('task_id', random_id())
        prompt = args['prompt']
        description = args['description']
        conversation = session.get_conversation(args['conversation_id'])
        nodes = figure_out_nodes_of_conversation()

        if type(nodes[-1]) == AssistantMessage:
            parts = [part for parts in node[-1].parts if type(part) in {TextContent}] # in the future we can add Image, etc
            return ExecutionResult(content=parts)
        else:
            return ExecutionResult(content=[
                TextPart(text=f"""The subagent finished successfully, here's a summary:
    {pretty_print_conversation(conversation)}
    """)
            ]) # obviously here it should be just a summary


```

When the LLM requests a subagent, it's just a tool call, so the whole tool execution flow happens naturally, it's all already covered.
The only special thing is that when the runner detects that a new `ToolExecution` completes, it looks at the name (for now) and if it matches `spawn_subagent` it kicks off the whole process of creating the `ChildConversation` and scheduling the runner and all that. To do so, it'll use the `metadata` provided by the tool. It also knows that whenever the `ChildConversation` finished it has to invoke the tool defined by the spawn_subagent tool call in the metadata attribute `process_subagent_result_tool_name`. So it's also transparent and encapsulated in the plugin behavior.

The important thing is to define the behavior. There's a contract between the "outside world" (contrib) and the runner. The runner knows how to implement things that are controlled by the outside world (by looking at the function name and the metadata). This allows developers to customize their "spawn subagent" and "process subagent result" functions without touching the core. The core's responsibility is to only schedule and run the things.

What's important is that if the `subagents_enabled` is False the ToolSpec should not be available to the LLM, so that means that the runner has to filter out any tools that are named `spawn_subagents`, and the same applies if `subagents_max_depth` has been reached (which by default will be 1).
There's a tiny detail here if the function `process_subagent_result_tool_name` should be available in `get_tools`. I'll leave it for now, I'll just add a description to the tool for the LLM with "do not invoke this tool" and in the future I'll add the concept of "private" tools (so they're not projected to LLMs but they're visible to the runtime).

## Projectors and Adapters

This is also something to keep simple. The important part is that the projector and adapter are fairly well architected and decoupled, so they have to know how to project a subagent conversation that makes sense. If the conversation has status COMPLETED and an ExecutionResult, they'll use that as a user message. If not they can just avoid it or mark something like "subagent failed" or "subagent didn't produce any results". This is something that can be improved in a future version but once the architecture is created and is sound.

### Tests

All these new tests should live in `tests/agent/subagents/`. As by default the `subagents_enabled` value is False the other tests should be unaffected and this new functionality should live in its own testing submodule

---

There are a few more things to review, like which events we'll provide and if it's necessary to create any new middleware endpoints, but I don't want to mess with that for now, it's something to worry about once we have validated the data model, spec, behavior and architecture.

Review my proposal, audit the code and tell me if there's something fundamentally wrong or broken with it. If there's nothing big, don't say anything, don't waste tokens for the sake of replying.

If you're going to tell me anything: feedback, issues, questions, etc. Treat me as your senior architect. I'm not familiar with the particularities of the codebase, so you have to give me some background context first, and then explain whatever you want to say. And you have just 1 minute of my attention, so you have to cut to the chase and be brief.
