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

`Models()` fetches the fixed models.dev catalogue once during initialization and caches it internally. Applications do not load or construct a catalogue.

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
models = Models()

authorization = await models.sign_in("openai")
# The host uses authorization.url to open the authorization page for the user.
await authorization.complete()

model = models.chat("openai/gpt-5", authorization=authorization)
```

The login flow and token refresh are provider-owned. Each authorization is independent, so the same `Models` instance can serve different users. The host decides whether and how to persist a user's authorization values.

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

## Usage values

Normalized usage is grouped by measurement type:

```python
usage.tokens.input_tokens
usage.tokens.reasoning_tokens
usage.cache.cache_read_tokens
usage.audio.output_audio_tokens
usage.cost.cost_usd
```

`UsageLedger` owns accumulation when an application records multiple responses.

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
