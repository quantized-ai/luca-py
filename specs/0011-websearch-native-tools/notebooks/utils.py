import os
import dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

ant_headers = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01"
}

def preview(obj, max_length=50, key_names=None, whitelist_keys=None):
    key_names = set(key_names or [])
    whitelist_keys = set(whitelist_keys or [])

    def walk(x):
        if isinstance(x, dict):
            
            return {
                k: (
                    (v if isinstance(v, str) and len(v) <= max_length else v[:max_length] + "<redacted for length> ...")
                    if isinstance(v, str) and (k not in whitelist_keys)
                    
                    # (key_names is None or k in key_names)
                    else walk(v)
                )
                for k, v in x.items()
            }

        if isinstance(x, list):
            return [walk(v) for v in x]

        if isinstance(x, tuple):
            return tuple(walk(v) for v in x)

        if isinstance(x, set):
            return {walk(v) for v in x}

        if isinstance(x, frozenset):
            return frozenset(walk(v) for v in x)

        return x

    return walk(obj)