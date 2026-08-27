# Models Provider

Models Provider gives applications one independent interface for discovering models,
resolving credentials, creating chat models, and collecting usage. It has no dependency
on LangMesh or Teacher.

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
before this call. An explicit credential store takes precedence over that snapshot.
OAuth tokens are stored and refreshed through the same credential store.

```python
from models_provider import ApiKeyCredential, CredentialStore, Models

credentials = CredentialStore.from_mapping({
    "openai": ApiKeyCredential("application-api-key"),
})

models = Models(credentials=credentials)
model = models.chat("openai/gpt-4.1-mini")
```

The credential interface is abstract and embedding applications provide persistent
implementations when required:

```python
class CredentialStore(ABC):
    def load(self, provider: str) -> Credential | None: ...
    def save(self, provider: str, credential: Credential) -> None: ...
    def clear(self, provider: str) -> None: ...
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

| Responsibility                          | Owner                 |
| --------------------------------------- | --------------------- |
| Model catalogue                         | Models Provider       |
| API keys, OAuth, and cloud credentials  | Models Provider       |
| Provider-specific clients               | Models Provider       |
| Sessions, tools, permissions, and files | Embedding application |
| Lesson generation                       | Teacher               |
