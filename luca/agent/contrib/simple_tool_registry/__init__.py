"""luca.agent.contrib.simple_tool_registry — the batteries-included registry.

The core framework knows only the four-method `ToolRegistry` contract. This
package supplies the standard implementations: `SimpleToolRegistry` (a static
tool list gated by one `PermissionPolicy`), `ProxyToolRegistry` (composition
+ routing over child registries), and the `PermissionPolicy` strategy
contract with `YoloPermissionPolicy` (allow everything).
"""

from .permissions import PermissionPolicy, YoloPermissionPolicy
from .registry import ProxyToolRegistry, SimpleToolRegistry

__all__ = [
    "PermissionPolicy",
    "ProxyToolRegistry",
    "SimpleToolRegistry",
    "YoloPermissionPolicy",
]
