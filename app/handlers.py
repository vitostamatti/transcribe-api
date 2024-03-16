from threading import Lock
from fastapi import HTTPException
import numpy as np
from transcribe import (
    Transcriber,
    Diarizer,
    diarize_transcript,
)
from .schemas import (
    AsrResponse,
    AsrSegmentResponse,
    TranscriptionResponse,
    TranscriptionSegmentResponse,
    DiarizationResponse,
    DiarizeSegmentResponse,
)


def transcribe_audio(
    audio, lock: Lock, transcriber: Transcriber
) -> TranscriptionResponse:
    with lock:
        try:
            transcript = transcriber.transcribe(audio)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error while running transcription"
            )

    response = TranscriptionResponse(
        transcription=[
            TranscriptionSegmentResponse(start=s.start, end=s.end, text=s.text)
            for s in transcript
        ]
    )
    return response


def diarize_audio(audio, lock: Lock, diarizer: Diarizer) -> DiarizationResponse:
    with lock:
        try:
            diarization = diarizer.diarize(audio)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error while running diarization"
            )
    response = DiarizationResponse(
        diarization=[
            DiarizeSegmentResponse(start=s.start, end=s.end, speaker=s.speaker)
            for s in diarization
        ]
    )
    return response


def asr_audio(
    audio: np.ndarray,
    transcriber_lock: Lock,
    transcriber: Transcriber,
    diarizer_lock: Lock,
    diarizer: Diarizer,
) -> AsrResponse:
    with transcriber_lock:
        try:
            transcript = transcriber.transcribe(audio)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error while running transcription"
            )
    with diarizer_lock:
        try:
            diarization = diarizer.diarize(audio)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error while running diarization"
            )
    try:
        diarized_transcript = diarize_transcript(transcript, diarization)
    except:
        raise HTTPException(
            status_code=500, detail="Error while parsing transcription and diarization"
        )

    response = AsrResponse(
        asr=[
            AsrSegmentResponse(start=s.start, end=s.end, text=s.text, speaker=s.speaker)
            for s in diarized_transcript
        ]
    )
    return response
