# models-provider

`models-provider` is an independent, transport-neutral boundary between an
application and the chat model it uses. It does not import or know about
LangMesh, Teacher, or any other application.

## Explicit API contract

Input:

```python
ModelConfiguration(
    provider: str,
    model: str,
    reasoning_effort: str | None = "high",
    temperature: float = 0.0,
    timeout_seconds: float | None = 300.0,
    context_length: int = 0,
    extra: Mapping[str, Any] = {},
)
```

Output:

```python
ModelProvider.create(configuration: ModelConfiguration) -> BaseChatModel
ModelProvider.scope() -> context manager
```

`scope()` is a neutral lifecycle hook for implementations that need temporary
request-local state. The contract carries no workspace, run identity, or
credential fields. `ProviderRegistry` is the built-in implementation for
application-owned factories.

```python
from models_provider import ModelConfiguration, ProviderRegistry

registry = ProviderRegistry()
registry.register(
    "my-provider",
    lambda configuration: make_model(configuration.model),
)
model = registry.create(ModelConfiguration(provider="my-provider", model="demo"))
with registry.scope():
    answer = model.invoke("Explain this paragraph.")
```

An integration library such as LangMesh may implement this contract in its own
repository. Applications then install both independent libraries and pass that
implementation into their application code.
