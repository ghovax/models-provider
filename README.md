# models-provider

`models-provider` is the independent model catalogue and construction layer. It turns
the public [models.dev](https://models.dev) data into typed provider and model records,
then routes a model choice to a concrete implementation. It has no dependency on
LangMesh or Teacher.

The catalogue is data, not application state. Loading it is explicit, so a library
embedding this package can refresh it at startup, cache it locally, or pass a
deterministic payload during a fast test.

## Explicit API contract

Catalogue input:

```python
from models_provider import ModelCatalogue, load_catalogue

catalogue = load_catalogue()  # explicit network call to models.dev/api.json
# Or: catalogue = ModelCatalogue.from_payload(saved_models_dev_json)
catalogue.models("openai")       # tuple[ModelRecord, ...]
catalogue.find("openai/gpt-5.4")  # ModelRecord | None
```

`ModelRecord` contains the provider-qualified identifier, model name and description,
reasoning/tool/attachment capabilities, input and output modalities, context and output
limits, pricing, dates, status, and the original provider/model fields in `extra`.
`ProviderRecord` contains the provider id, name, SDK family, credential
environment-variable names, documentation URL, and default API endpoint.

Model construction input and output:

```python
from models_provider import LiteLLMProvider, ModelConfiguration

provider = LiteLLMProvider(
    catalogue,
    api_keys={"openai": "..."},       # supplied by the embedding application
    api_bases={},
)
model = provider.create(
    ModelConfiguration(
        provider="openai",
        model="gpt-5.4",
        reasoning_effort="high",
        temperature=0.0,
        timeout_seconds=120.0,
    )
)
answer = model.invoke("Explain this paragraph.")
```

The input to `create` is only a `ModelConfiguration`: provider id, model id, and request
defaults. Working directories, session ids, run ids, transcripts, and sources do not
belong here. The output is a LangChain `BaseChatModel`; callers use its normal `invoke`,
`ainvoke`, `stream`, `astream`, and tool-binding methods. `LiteLLMProvider` is the
built-in generic implementation for the hosted providers represented in models.dev.

Usage totals use the shared `ModelUsage` value from this package. Applications can
aggregate those values with `combined_with` and expose them by the model identifier they
used.

For applications that need a different transport or an OAuth-specific model,
`ProviderRegistry` is the extension point:

```python
from models_provider import ModelConfiguration, ModelRecord, ProviderRegistry
from models_provider import LiteLLMChatModel

registry = ProviderRegistry(catalogue)

def build_local_openai_compatible_model(
    configuration: ModelConfiguration, record: ModelRecord | None
):
    return LiteLLMChatModel(
        model=f"openai/{configuration.model}",
        api_base="http://127.0.0.1:11434/v1",
        temperature=configuration.temperature,
        timeout=configuration.timeout_seconds,
        context_length=record.context_length if record else 0,
    )

registry.register("custom", build_local_openai_compatible_model)
model = registry.create(
    ModelConfiguration(provider="custom", model="llama3.1:8b")
)
```

The factory receives the selected `ModelRecord | None`, allowing an implementation to
use models.dev capabilities without reimplementing parsing. Credentials and transport
settings belong to the provider instance, never to the model identifier or to unrelated
application workflows.
