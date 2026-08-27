from src.inference.postprocessing import postprocess_response
from src.inference.schema import InferenceResponse


def _response(generated_text: str) -> InferenceResponse:
    return InferenceResponse(
        request_id="req-1",
        generated_text=generated_text,
        model_version="v1",
        input_token_count=10,
        output_token_count=5,
        latency_seconds=0.1,
    )


def test_postprocess_strips_surrounding_whitespace():
    response = postprocess_response(_response("  a cat on a mat \n"))
    assert response.generated_text == "a cat on a mat"


def test_postprocess_preserves_interior_content():
    response = postprocess_response(_response("line one\nline two"))
    assert response.generated_text == "line one\nline two"


def test_postprocess_returns_new_object_without_mutating_input():
    original = _response("  padded text  ")
    result = postprocess_response(original)

    assert original.generated_text == "  padded text  "
    assert result.generated_text == "padded text"
    assert result is not original


def test_postprocess_is_a_noop_for_already_clean_text():
    original = _response("already clean")
    result = postprocess_response(original)

    assert result is original
