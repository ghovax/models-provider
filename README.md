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

## Authentication contract

Models Provider owns provider credential resolution. An embedding application supplies
only a credential store and optional in-memory overrides:

```python
import webbrowser

from models_provider import (
    MemoryCredentialStore,
    ProviderAuthentication,
    load_catalogue,
)

catalogue = load_catalogue()
credential_store = MemoryCredentialStore()
authentication = ProviderAuthentication(
    catalogue=catalogue,
    api_keys={"openai": "key-from-the-application"},
    store=credential_store,
)

resolved = authentication.resolve("openai")
# ApiKeyResolution(provider="openai", api_key="...", api_base="", source="configured")
status = authentication.status("openai")
# AuthenticationStatus(provider="openai", method="api_key", signed_in=True, source="configured")
```

Resolution checks explicit keys, stored `ApiKeyCredential` values, the provider's
declared environment variables, and finally an anonymous provider key when one is
declared. Cloud profiles such as Vertex and Bedrock return named environment values
without mislabeling project or region settings as API keys. `save_api_key` and
`sign_out` persist or remove credentials through the supplied store. The library never
writes a secret file by itself.

The package also provides a reusable OAuth contract for providers that publish standard
authorization endpoints:

```python
from models_provider import OAuthConfiguration

authentication.register_oauth(
    "example-provider",
    OAuthConfiguration(
        authorization_url="https://login.example.com/authorize",
        token_url="https://login.example.com/token",
        client_id="registered-client-id",
        scopes=("inference",),
        redirect_uri="http://127.0.0.1:8765/callback",
    ),
)
async def sign_in():
    flow = authentication.flow("example-provider")
    await flow.start()
    webbrowser.open(flow.authorize_url)
    return await flow.wait()
```

`OAuthLoginFlow` implements state validation, PKCE, callback handling, token exchange,
and persistence. `DeviceLoginFlow` is selected when the configuration includes a device
authorization endpoint. Refresh and authenticated request headers are handled by the
same registered adapter. ChatGPT and Cursor use built-in adapters because their account
sign-in protocols are provider-specific; other providers can register either the
standard flow or a small custom adapter without importing LangMesh.

Authentication status is intentionally safe to publish: it reports the provider, method,
source, expiry, and display account only. Token and API-key values stay inside the
credential store and request material.
