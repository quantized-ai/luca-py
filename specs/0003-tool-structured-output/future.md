I am considering adding a few different types of "validations options":
- strict: IF (and only if) the tool produced `structured_content`, it's checked against the schema
- mandatory: the execution result MUST produce structured content, and it must validate against the schema
- warn: stores whatever is produced as structured_content but raises warnings on missing or non conforming schemas

If there's no validation option, the runner/tool/registry just store the structured_content without checking anything.

Now, I'm not sure if I should add the validation because it adds a lot of complexity that I'm not sure I want to handle, and I think might belong to the user. For example:

- Who's responsibility is this? on the one hand, the tool SHOULD know the type of validation required. But the responsibility to validate should be in the registry or runner?

Review what I want to do and give me your impressions about the whole Tool structured output thing first: be very brief, I'm pretty settled on this. If you don't have any pushbacks or anything important to add just move on.
And second, give me your impressions about validation: types, who owns it, who sees it, etc.

This is obviously a brainstorming, there's a lot to define, even if the current client supports it. but I don't care about these details for now, and you shouldn't either, we're discussing high level architectural changes.

```python
AgentSession(
    ...
    output_schema=CapitalOutput
)
```
