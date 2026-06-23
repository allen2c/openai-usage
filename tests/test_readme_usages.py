import agents
from openai.types.audio.transcription import (
    UsageDuration,
    UsageTokens,
    UsageTokensInputTokenDetails,
)
from openai.types.batch_usage import BatchUsage
from openai.types.batch_usage import InputTokensDetails as BatchInputTokensDetails
from openai.types.batch_usage import OutputTokensDetails as BatchOutputTokensDetails
from openai.types.beta.threads.run import Usage as RunUsage
from openai.types.beta.threads.runs.run_step import Usage as RunStepUsage
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from openai.types.images_response import Usage as ImagesUsage
from openai.types.images_response import (
    UsageInputTokensDetails as ImagesInputTokensDetails,
)
from openai.types.images_response import (
    UsageOutputTokensDetails as ImagesOutputTokensDetails,
)
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage
from openai.types.realtime.realtime_response_usage_input_token_details import (
    RealtimeResponseUsageInputTokenDetails,
)
from openai.types.realtime.realtime_response_usage_output_token_details import (
    RealtimeResponseUsageOutputTokenDetails,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from openai_usage import Usage


def test_usage_initialization():
    usage = Usage()
    assert usage.requests == 0
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_from_openai_completion_usage():
    openai_usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30


def test_from_openai_completion_usage_with_details():
    openai_usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=5),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=3),
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.output_tokens_details.reasoning_tokens == 3


def test_from_openai_response_usage():
    openai_usage = ResponseUsage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=InputTokensDetails(cached_tokens=5),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.output_tokens_details.reasoning_tokens == 8


def test_from_openai_realtime_usage():
    openai_usage = RealtimeResponseUsage(
        input_tokens=100,
        output_tokens=200,
        total_tokens=300,
        input_token_details=RealtimeResponseUsageInputTokenDetails(
            audio_tokens=60, text_tokens=40, cached_tokens=10
        ),
        output_token_details=RealtimeResponseUsageOutputTokenDetails(
            audio_tokens=150, text_tokens=50
        ),
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 100
    assert usage.output_tokens == 200
    assert usage.total_tokens == 300
    assert usage.input_tokens_details.cached_tokens == 10
    assert usage.input_tokens_details.audio_tokens == 60
    assert usage.input_tokens_details.text_tokens == 40
    assert usage.output_tokens_details.audio_tokens == 150
    assert usage.output_tokens_details.text_tokens == 50
    assert usage.output_tokens_details.reasoning_tokens == 0


def test_realtime_usage_backward_compatible_json():
    # Old serialized payloads (no audio/text fields) must still validate.
    old_json = (
        '{"requests":1,"input_tokens":10,"output_tokens":20,"total_tokens":30,'
        '"input_tokens_details":{"cached_tokens":5},'
        '"output_tokens_details":{"reasoning_tokens":3}}'
    )
    usage = Usage.model_validate_json(old_json)
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.input_tokens_details.audio_tokens == 0
    assert usage.output_tokens_details.reasoning_tokens == 3
    assert usage.output_tokens_details.audio_tokens == 0


def test_add_usage_accumulates_audio_tokens():
    usage = Usage.from_openai(
        RealtimeResponseUsage(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_token_details=RealtimeResponseUsageInputTokenDetails(
                audio_tokens=60, cached_tokens=10
            ),
            output_token_details=RealtimeResponseUsageOutputTokenDetails(
                audio_tokens=150
            ),
        )
    )
    usage.add(usage.model_copy(deep=True))
    assert usage.input_tokens_details.audio_tokens == 120
    assert usage.input_tokens_details.cached_tokens == 20
    assert usage.output_tokens_details.audio_tokens == 300


def test_from_openai_batch_usage():
    openai_usage = BatchUsage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=BatchInputTokensDetails(cached_tokens=4),
        output_tokens_details=BatchOutputTokensDetails(reasoning_tokens=6),
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.input_tokens_details.cached_tokens == 4
    assert usage.output_tokens_details.reasoning_tokens == 6


def test_from_openai_embedding_usage():
    usage = Usage.from_openai(EmbeddingUsage(prompt_tokens=15, total_tokens=15))
    assert usage.requests == 1
    assert usage.input_tokens == 15
    assert usage.output_tokens == 0
    assert usage.total_tokens == 15


def test_from_openai_run_and_run_step_usage():
    run = Usage.from_openai(
        RunUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    )
    assert run.input_tokens == 100
    assert run.output_tokens == 50
    step = Usage.from_openai(
        RunStepUsage(prompt_tokens=8, completion_tokens=2, total_tokens=10)
    )
    assert step.input_tokens == 8
    assert step.output_tokens == 2


def test_from_openai_images_usage():
    openai_usage = ImagesUsage(
        input_tokens=30,
        output_tokens=70,
        total_tokens=100,
        input_tokens_details=ImagesInputTokensDetails(image_tokens=10, text_tokens=20),
        output_tokens_details=ImagesOutputTokensDetails(image_tokens=70, text_tokens=0),
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens_details.image_tokens == 10
    assert usage.input_tokens_details.text_tokens == 20
    assert usage.output_tokens_details.image_tokens == 70


def test_from_openai_transcription_tokens_usage():
    openai_usage = UsageTokens(
        type="tokens",
        input_tokens=40,
        output_tokens=10,
        total_tokens=50,
        input_token_details=UsageTokensInputTokenDetails(
            audio_tokens=35, text_tokens=5
        ),
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 40
    assert usage.input_tokens_details.audio_tokens == 35
    assert usage.output_tokens == 10


def test_from_openai_transcription_duration_usage():
    usage = Usage.from_openai(UsageDuration(type="duration", seconds=12.5))
    assert usage.requests == 1
    assert usage.seconds == 12.5
    assert usage.input_tokens == 0
    assert usage.total_tokens == 0


def test_add_usage_accumulates_seconds_and_image_tokens():
    usage = Usage.from_openai(
        ImagesUsage(
            input_tokens=30,
            output_tokens=70,
            total_tokens=100,
            input_tokens_details=ImagesInputTokensDetails(
                image_tokens=10, text_tokens=20
            ),
            output_tokens_details=ImagesOutputTokensDetails(
                image_tokens=70, text_tokens=0
            ),
        )
    )
    usage.seconds = 5.0
    usage.add(usage.model_copy(deep=True))
    assert usage.input_tokens_details.image_tokens == 20
    assert usage.output_tokens_details.image_tokens == 140
    assert usage.seconds == 10.0


def test_from_openai_inplace():
    openai_usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    usage = Usage.from_openai(openai_usage, inplace=True)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30

    usage.model = "gpt-4o"
    usage.cost = usage.estimate_cost()
    print(f"cost: {round(float(usage.cost), 6)} USD")


def test_add_usage():
    usage1 = Usage(requests=1, input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = Usage(requests=2, input_tokens=15, output_tokens=25, total_tokens=40)
    usage1.add(usage2)
    assert usage1.requests == 3
    assert usage1.input_tokens == 25
    assert usage1.output_tokens == 45
    assert usage1.total_tokens == 70


def test_from_agents_usage():
    agents_usage = agents.Usage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=InputTokensDetails(cached_tokens=5),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
    )
    usage = Usage.from_openai(agents_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.output_tokens_details.reasoning_tokens == 8


def test_from_agents_run_context_wrapper():
    agents_usage = agents.Usage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=InputTokensDetails(cached_tokens=5),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
    )
    context_wrapper = agents.RunContextWrapper(context=None, usage=agents_usage)
    usage = Usage.from_openai(context_wrapper)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.output_tokens_details.reasoning_tokens == 8


def test_estimate_cost():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)
    cost = usage.estimate_cost_str("gpt-4o-mini")
    assert isinstance(cost, str)
    assert float(cost) > 0


def test_estimate_cost_realtime():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)
    cost = usage.estimate_cost_str("anthropic/claude-3-haiku", realtime_pricing=True)
    assert isinstance(cost, str)
    assert float(cost) > 0


def test_estimate_cost_no_model():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)
    cost = usage.estimate_cost_str()
    assert isinstance(cost, str)
    assert float(cost) > 0


def test_estimate_cost_not_found():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)
    assert usage.estimate_cost("not-a-real-model") == 0.0
