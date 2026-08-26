---
name: models-provider
description: Select and configure interchangeable chat-model providers through models-provider.
---

# Model providers

Ask for the provider, model, reasoning effort, working directory, and explicit credential source before building a model.

Use `ModelConfiguration` for the model choice and `LangMeshProvider` when the application should use LangMesh's provider catalogue, API-key configuration, custom endpoints, or ChatGPT/Cursor credential stores. Keep credentials in the caller-owned store or explicit provider configuration; do not invent environment variables.

Use `ProviderRegistry` when an application owns its model implementations. Register one factory per provider and pass the registry anywhere a `ModelProvider` is accepted.

Activate native credential state only for the call scope with `with provider.scope():`. Keep long-running runs detached and persist their outputs and checkpoints before the process exits.
