"""Service layer for inference and landmark processing."""

from backend.services.inference_service import ISLInferenceService, get_inference_service

__all__ = ["ISLInferenceService", "get_inference_service"]
