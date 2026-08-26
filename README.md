# models-provider

`models-provider` is the transport-neutral boundary between an application and
the chat model it uses. Applications choose a provider and model through
`ModelConfiguration`; the provider owns credentials, transport, caching, and
provider-specific setup.

## Explicit API contract

Input to a provider:

```python
ModelConfiguration(
    provider: str,
    model: str,
    reasoning_effort: str | None = "high",
    temperature: float = 0.0,
    timeout_seconds: float | None = 300.0,
    context_length: int = 0,
    session_id: str = "",
    extra: Mapping[str, Any] = {},
)
```

`ModelConfiguration.identifier` is the stable `"provider/model"` string.
`working_directory: Path` and an optional `session_id: str` are supplied to the
provider call. Credentials are not fields of this value.

Output from a provider:

```python
ModelProvider.create(
    configuration: ModelConfiguration,
    *,
    working_directory: Path,
    session_id: str = "",
) -> langchain_core.language_models.BaseChatModel
```

`ModelProvider.scope() -> context manager` activates any request-local provider
state. `ProviderRegistry` implements the same contract for application-owned
factories. `LangMeshProvider` adapts LangMesh's existing model catalogue and
credential stores to this boundary.

```bash
uv add "models-provider @ git+https://github.com/ghovax/models-provider.git"
uv add "models-provider[langmesh] @ git+https://github.com/ghovax/models-provider.git"
```

```python
from pathlib import Path

from models_provider import LangMeshProvider, ModelConfiguration

provider = LangMeshProvider(providers={"anthropic": "sk-ant-..."})
model = provider.create(
    ModelConfiguration(provider="anthropic", model="claude-sonnet-4-5"),
    working_directory=Path.cwd(),
)
with provider.scope():
    answer = model.invoke("Explain this paragraph.")
```

For an application-owned backend, register a factory:

```python
from models_provider import ModelConfiguration, ProviderRegistry

registry = ProviderRegistry()
registry.register(
    "my-provider",
    lambda configuration, directory, session: make_model(configuration.model),
)
```
