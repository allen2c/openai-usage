import pydantic


class Pricing(pydantic.BaseModel):
    model: str
    input: float
    cached_input: float | None = None
    output: float
