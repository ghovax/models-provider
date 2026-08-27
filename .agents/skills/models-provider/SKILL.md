---
name: models-provider
description: Select and configure interchangeable chat-model providers through models-provider.
---

# Model providers

Use `Models` for model discovery, authentication, and construction:

```python
from models_provider import Models

models = Models.from_environment()
model = models.chat("openai/gpt-4.1-mini")
```

`Models()` does not inspect process environment variables. Use
`Models.from_environment()` to explicitly capture them, or provide an injected
`CredentialStore`. If both are provided, the credential store wins.
OAuth authorization returns a URL-bearing handle; the host decides whether to display or
open the URL and then calls `complete()`.

Keep Models Provider independent from application runtimes. Working directories, session
identities, tools, checkpoints, and lesson inputs do not belong in this package.
