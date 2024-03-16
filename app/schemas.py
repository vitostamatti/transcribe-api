# define output schema
from pydantic import BaseModel


class TranscriptionSegmentResponse(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    transcription: list[TranscriptionSegmentResponse]


class DiarizeSegmentResponse(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizationResponse(BaseModel):
    diarization: list[DiarizeSegmentResponse]


class AsrSegmentResponse(BaseModel):
    start: float
    end: float
    text: str
    speaker: str


class AsrResponse(BaseModel):
    asr: list[AsrSegmentResponse]
