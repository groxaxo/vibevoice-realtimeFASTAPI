"""Explicit error classes for runner operations."""


class UnknownModelError(Exception):
    """Raised when the requested model key cannot be resolved."""

    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        super().__init__(
            f"Unknown model: '{model_key}'. "
            "Use GET /v1/models or GET /config to see available models and aliases."
        )


class CapabilityError(Exception):
    """Raised when an operation is not supported by the selected model."""

    def __init__(self, model_key: str, capability: str) -> None:
        self.model_key = model_key
        self.capability = capability
        super().__init__(f"Model '{model_key}' does not support '{capability}'.")


class BackendUnavailableError(Exception):
    """Raised when a model's runtime backend is not installed or configured."""

    def __init__(self, model_key: str, detail: str | None = None) -> None:
        self.model_key = model_key
        msg = f"Model '{model_key}' is registered but its runtime backend is unavailable."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class InvalidRequestForModelError(Exception):
    """Raised when request parameters are incompatible with a model family."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
