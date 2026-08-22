import json
import httpx
import sys
from utils import *

messages = [{
    "role": "user",
    "content": "What's the stock price of NVIDIA today?"
}]

web_search_tool = {
    "type": "web_search_20260318",
    "name": "web_search",
    "max_uses": 5,
}
if "--direct" in sys.argv:
    web_search_tool["allowed_callers"] = ["direct"]

resp = httpx.post(
    "https://api.anthropic.com/v1/messages",
    headers=ant_headers,
    timeout=120.0,
    json={
        "model": "claude-sonnet-5",
        "max_tokens": (4096 * 10),
        "output_config": {
            "effort": "low"
        },
        "messages": messages,
        "tools": [web_search_tool]
    }
)

# resp.raise_for_status()
# print(resp)
# print(resp.json())
if resp.status_code != 200:
    print(resp)
    print(resp.json())
    exit()

import json

print(json.dumps(preview(resp.json(), 50, whitelist_keys=["text"]), indent=2))

print("Stop Reason:")
print(resp.json()['stop_reason'])

print("Usage:")
print(resp.json()['usage'])

print("Citations:")
print(json.dumps([
    citation
    for block in resp.json()["content"]
    for citation in block.get("citations", [])
], indent=2))
