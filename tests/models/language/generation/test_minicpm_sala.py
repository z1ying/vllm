# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HF ↔ vLLM parity tests for MiniCPM-SALA model.

Focused on token-level alignment between Hugging Face and vLLM implementations.
"""

import importlib

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.platforms import current_platform

from ...utils import check_logprobs_close

MODEL = "openbmb/MiniCPM-SALA"


def get_attention_backend():
    try:
        return "TRITON_ATTN" if current_platform.is_rocm() else "auto"
    except (AttributeError, TypeError):
        return "auto"


ATTN_BACKEND = get_attention_backend()
MAX_NUM_SEQS = 4

# Skip all tests if fla is not available (HF model requirement)
try:
    importlib.import_module("fla")
except ImportError:
    pytest.skip("fla package not installed - required for MiniCPM-SALA tests. Run: pip install flash-linear-attention", allow_module_level=True)


@pytest.fixture(scope="module")
def example_prompts():
    return [
        "Hello, my name is",
        "The capital of France is",
        "What is the meaning of life?",
        "Explain quantum computing in simple terms",
    ]


def _check_model_availability():
    try:
        importlib.import_module("fla")
    except ImportError:
        pytest.skip("fla package not installed - required for HF MiniCPM-SALA model. Run: pip install flash-linear-attention")
    
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pytest.skip(f"Model {MODEL} not available or transformers version incompatible")


def _skip_if_no_gpu():
    try:
        device_type = current_platform.device_type
        if not device_type:
            pytest.skip("Device type detection failed - likely no GPU available")
    except (AttributeError, TypeError):
        pytest.skip("Device type detection failed - likely no GPU available")


@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_hf_vllm_parity_greedy(
    hf_runner,
    vllm_runner,
    example_prompts,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Test HF vs vLLM parity with greedy decoding and logprobs comparison."""
    _check_model_availability()
    _skip_if_no_gpu()

    with hf_runner(MODEL) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        MODEL, max_num_seqs=MAX_NUM_SEQS, attention_backend=ATTN_BACKEND
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [10])
def test_hf_vllm_parity_long_context(
    hf_runner,
    vllm_runner,
    example_prompts,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Test HF vs vLLM parity with longer context to test lightning attention state."""
    _check_model_availability()
    _skip_if_no_gpu()

    long_prompts = [p * 10 for p in example_prompts[:2]]

    with hf_runner(MODEL) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            long_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        MODEL, max_num_seqs=MAX_NUM_SEQS, attention_backend=ATTN_BACKEND
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            long_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
