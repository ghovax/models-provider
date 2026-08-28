# Models Provider

Models Provider gives applications one independent interface for discovering models,
resolving credentials, creating chat models, and collecting usage. It uses the public
[models.dev catalogue](https://models.dev) and has no dependency on any application
library.

## Public flow

```python
from models_provider import Models

models = Models.from_environment()

model = models.chat(
    "openai/gpt-4.1-mini",
    temperature=0.0,
)

answer = model.invoke("Explain spaced repetition in two sentences.")
```

`Models` loads the models.dev catalogue lazily on first use. Applications do not need to
load or pass a catalogue explicitly.

```python
models.list()
models.list("openai")
models.find("openai/gpt-4.1-mini")
```

## Credentials

`Models()` never reads process environment variables. Call `Models.from_environment()`
when that is the intended credential source; it captures the environment explicitly at
construction time. The library does not parse `.env` files; the host must load them
before this call. When both are supplied, an explicit credential store takes precedence
over that environment snapshot. OAuth tokens are stored and refreshed through the
selected credential store.

```python
from models_provider import ApiKeyCredential, CredentialStore, Models

credentials = CredentialStore.from_mapping(
    {
        "openai": ApiKeyCredential("sk-proj-...7Qx2"),
    }
)

models = Models(credentials=credentials)
model = models.chat("openai/gpt-4.1-mini")
```

The credential interface is abstract. Embedding applications provide persistent
implementations when required; stores hold `ApiKeyCredential`, `EnvironmentCredential`,
or provider-specific OAuth token values:

```python
class CredentialStore(ABC):
    def load(self, provider_identifier: str) -> object | None: ...
    def save(self, provider_identifier: str, credentials: object) -> None: ...
    def clear(self, provider_identifier: str) -> None: ...
```

`InMemoryCredentialStore` is a concrete store for short-lived applications and mock
runs. It is not the credential abstraction.

## OAuth

```python
authorization = await models.sign_in("chatgpt")

print(authorization.url)
# The host decides whether to display, copy, or open the URL.

await authorization.complete()
model = models.chat("chatgpt/gpt-5")
```

The library prepares the callback listener and returns the URL. It does not open a
browser or make a user-interface decision.

Hosts can ask the provider for the redirect URI registered for its OAuth client. The
host must keep its `state` and `code_verifier` until the callback, validate the returned
state, and then exchange the one-time code:

```python
from models_provider import ProviderAuthentication

authentication = ProviderAuthentication()
redirect_uri = authentication.redirect_uri("chatgpt")
authorization = authentication.authorization_request(
    "chatgpt",
    redirect_uri,
)
print(authorization.authorize_url)

# In the callback handler, after checking that the state matches:
tokens = await authorization.exchange(code)
```

For ChatGPT, `redirect_uri` is the registered loopback URI
`http://localhost:1455/auth/callback`. A host without a local listener can display the
URL, let the browser return to localhost, and receive the copied one-time code through
its own completion endpoint. Other providers may return a registered HTTPS callback
instead.

The host can serialize credentials before persisting them and deserialize them when
restoring them:

```python
payload = authentication.serialize_token("chatgpt", tokens)
restored_tokens = authentication.deserialize_token("chatgpt", payload)
```

The host chooses where to persist the payload. Providers own their token shape, refresh
behavior, and request headers.

## Model contract

```python
ModelProvider.chat(
    model_identifier="provider/model",
    temperature=0.0,
    reasoning_effort="high",
    timeout_seconds=300.0,
) -> BaseChatModel
```

The model identifier is the only model-selection value callers need. Catalogue records,
provider routing, authentication headers, refresh behavior, and usage normalization
remain inside Models Provider.

## Ownership

Models Provider owns:

- the models.dev catalogue;
- credential resolution and provider-specific authentication;
- provider-specific clients, authentication headers, and refresh behavior;
- usage normalization.

The embedding application owns:

- the credential store and its persistence policy;
- sessions, tools, permissions, and files;
- application workflows and domain behavior.

Credentials remain in the store supplied by the embedding application; Models Provider
does not choose a storage backend or write secret files by itself.
