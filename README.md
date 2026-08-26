# models-provider

`models-provider` is the small boundary between an application and the chat models it uses. A caller supplies a provider/model choice and receives a LangChain `BaseChatModel`; credentials, transport, caching, and persistence stay behind the provider implementation.

The package includes a `ProviderRegistry` for application-owned providers and a `LangMeshProvider` adapter. The latter uses LangMesh's existing provider catalogue, API-key configuration, custom endpoints, and ChatGPT/Cursor credential stores without making those details part of the application graph.

```bash
uv add "models-provider @ git+https://github.com/ghovax/models-provider.git"
uv add "models-provider[langmesh] @ git+https://github.com/ghovax/models-provider.git"
```

```python
from pathlib import Path

from models_provider import LangMeshProvider, ModelSpec

provider = LangMeshProvider(providers={"anthropic": "sk-ant-..."})
model = provider.create(
    ModelSpec(provider="anthropic", model="claude-sonnet-4-5"),
    working_directory=Path.cwd(),
)
```

Native LangMesh credentials are activated only around the calls that use them:

```python
with provider.scope():
    answer = model.invoke("Explain this paragraph.")
```

For an application-owned backend, register a factory instead of depending on LangMesh:

```python
from models_provider import ModelSpec, ProviderRegistry

registry = ProviderRegistry()
registry.register("my-provider", lambda spec, directory, session: make_model(spec.model))
```

No environment variables are required by this boundary. Pass configuration explicitly, and keep long-running application work detached with its own durable output and checkpoint locations.
