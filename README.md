# Models Provider

Models Provider gives applications one interface for selecting models, resolving provider access, creating chat models, and collecting usage. Model metadata is loaded privately from the public [models.dev catalogue](https://models.dev).

## Public flow

Pass provider values directly to `Models`, then select a provider-qualified model:

```python
from models_provider import Models

models = Models({
    "openai": "OPENAI_API_KEY",
})

model = models.chat(
    "openai/gpt-4.1-mini",
    temperature=0.2,
    top_p=0.9,
)

answer = model.invoke("Explain spaced repetition in two sentences.")
```

The catalogue is fetched lazily on the first lookup and cached internally. Applications do not load or construct a catalogue.

The model identifier describes the model publisher, not the authentication mechanism. The access implementation is selected internally from the provider and the supplied values. For example, `openai/gpt-5` remains the model identifier when the available access is an API key or an OpenAI account session.

## Provider values

The constructor accepts one ordinary dictionary. Values can be literal credentials, environment-variable names, or provider-specific mappings:

```python
Models({"openai": "sk-proj-...7Qx2"})
```

```python
Models({"openai": "OPENAI_API_KEY"})
```

An uppercase environment-variable name is resolved from the process environment. Persistence is owned by the embedding application; Models Provider does not expose a credential-store abstraction or load credential files.

## OAuth

OAuth values use the same dictionary. The host controls how the authorization URL is displayed:

```python
values = {"openai": {}}
models = Models(values)

authorization = await models.sign_in("openai")
print(authorization.url)
await authorization.complete()

model = models.chat("openai/gpt-5")
```

The login flow and token refresh are provider-owned. Refreshed values live in the supplied dictionary for the lifetime of the process; the host decides whether and how to persist them.

## Model contract

```python
models.chat(
    "provider/model",
    temperature=0.2,
    top_p=0.9,
    reasoning_effort="high",
    max_output_tokens=512,
) -> BaseChatModel
```

Request settings are ordinary keyword arguments. The selected access implementation validates and translates them to its transport. There is no public options object and no provider-specific access class required from the caller.

## Ownership

Models Provider owns:

- the private models.dev catalogue and its built-in cache;
- provider and model selection;
- provider-specific authentication and transport;
- request-option normalization;
- usage normalization.

The embedding application owns:

- the provider-values dictionary;
- credential persistence, if needed;
- application workflows, sessions, tools, permissions, and files.
