---
name: models-provider
description: Select and configure interchangeable chat-model providers through models-provider.
---

# Model providers

Use `Models` for model discovery, authentication, and construction:

```python
from models_provider import Models

models = Models()
model = models.chat("openai/gpt-4.1-mini")
```

Credentials come from an injected `CredentialStore` or provider environment variables.
OAuth authorization returns a URL-bearing handle; the host decides whether to display or
open the URL and then calls `complete()`.

Keep Models Provider independent from application runtimes. Working directories, session
identities, tools, checkpoints, and lesson inputs do not belong in this package.
