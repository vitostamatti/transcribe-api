from threading import Lock
import time
from typing import BinaryIO
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse
import torch
import torchaudio
import torchaudio.functional as F
import uvicorn
from transcribe import (
    Transcriber,
    Diarizer,
)

from .schemas import AsrResponse, DiarizationResponse, TranscriptionResponse
from .config import Settings
from .handlers import diarize_audio, transcribe_audio, asr_audio
import os

settings = Settings()
os.environ["HF_TOKEN"] = settings.hf_token

app = FastAPI(
    title="Transcribe API",
    description="Rest API for the Transcribe package",
    version="0.1.0",
)

diarizer_lock = Lock()
diarizer = Diarizer()
transcriber_lock = Lock()
transcriber = Transcriber()


def encode_audio_binaries(binaries: BinaryIO):
    try:
        audio, sample_rate = torchaudio.load(binaries)
        return torch.mean(F.resample(audio, sample_rate, 16_000), dim=0).numpy()
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/transcribe", tags=["Endpoints"], description="Runs Transcription")
async def transcribe(
    audio_file: UploadFile,
) -> TranscriptionResponse:
    return transcribe_audio(
        audio=encode_audio_binaries(audio_file.file),
        lock=transcriber_lock,
        transcriber=transcriber,
    )


@app.post("/diarize", tags=["Endpoints"], description="Runs Diarization")
async def diarize(
    audio_file: UploadFile,
) -> DiarizationResponse:
    return diarize_audio(
        encode_audio_binaries(audio_file.file), lock=diarizer_lock, diarizer=diarizer
    )


@app.post("/asr", tags=["Endpoints"], description="Runs Transcription and Diarization")
async def asr(
    audio_file: UploadFile,
) -> AsrResponse:
    return asr_audio(
        audio=encode_audio_binaries(audio_file.file),
        transcriber_lock=transcriber_lock,
        transcriber=transcriber,
        diarizer_lock=diarizer_lock,
        diarizer=diarizer,
    )


@app.get("/heartbeat")
async def hearbeat():
    return int(time.time())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
