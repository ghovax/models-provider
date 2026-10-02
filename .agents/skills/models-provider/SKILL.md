---
name: models-provider
description: Select and configure interchangeable chat-model providers through models-provider.
---

# Model providers

Use `Models` for model discovery, authentication, and construction:

```python
from models_provider import Models

models = Models({"openai": {"api_key": "YOUR_API_KEY"}})
model = models.chat("openai/gpt-4.1-mini")
```

`Models()` does not inspect environment variables or parse configuration files. The host supplies provider values explicitly. Use `models.providers()` and `models.list()` to obtain typed metadata, and `model.bind_tools(...)` with `model.astream(...)` for streamed agent requests. OAuth authorization returns a URL-bearing handle; the host decides whether to display or open the URL and then calls `complete()`.

For OpenCode, pass `OpenCodeRequestContext(session_id=...)` using the `opencode_request_context` request option. Keep that identity stable for a conversation. This is transport metadata; the host owns the conversation lifecycle.

Keep Models Provider independent from application runtimes. Working directories, session identities, tools, checkpoints, and lesson inputs do not belong in this package.
