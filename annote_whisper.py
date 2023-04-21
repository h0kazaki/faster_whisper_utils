import os
import argparse
from typing import Optional

import ffmpeg
import tempfile

from faster_whisper import WhisperModel, decode_audio
from pyannote.audio import Audio, Pipeline

def convert_mp4_to_wav(input_path: str) -> str:
    """Convert mp4 to wav.

    Args:
        input_path (str): path to mp4 file.

    Returns:
        str: path to wav file.
    """

    # generate temporary file name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        output_file = f.name

    # ffmpeg command
    stream = ffmpeg.input(input_path)
    stream = ffmpeg.output(stream, output_file, format='wav')

    # run ffmpeg
    ffmpeg.run(stream)

    return output_file


def annote_whisper(audio_path: str,
                   language: Optional[str] = None,
                   model_name: str = 'large-v2',
                   num_speakers: Optional[int] = None,
                   ) -> None:
    """
    Annotate a file with speaker diarization and speech recognition.

    vars:
        audio_file: Path to audio file.
        language: Language to use for speech recognition.
        model_name: Model to use for speech recognition.
        num_speakers: Number of speakers to use for speaker diarization.
    """
    model = WhisperModel(model_name)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.environ["HF_TOKEN"])
    if audio_path.endswith('.mp4'):
        temp_path = convert_mp4_to_wav(audio_path)
        audio_path = temp_path

    try:
        diarization = pipeline(audio_path, num_speakers=num_speakers)

        audio = Audio(sample_rate=16000, mono=True)

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            waveform, sample_rate = audio.crop(audio_path, segment)
            sub_segments = model.transcribe(audio=waveform.squeeze().numpy(),
                                            language=language,
                                            )[0]
            text = ' '.join([sub_segment.text for sub_segment in sub_segments])
            print(f"[{segment.start:03.1f}s - {segment.end:03.1f}s] {speaker}: {text}")
    finally:
        if audio_path.endswith('.mp4'):
            os.remove(temp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate a file with speaker diarization and speech recognition.')
    parser.add_argument('audio_path', type=str, help='Path to audio file.')
    parser.add_argument('-l', '--language', type=str, default=None, help='Language to use for speech recognition. e.g. ja, en')
    parser.add_argument('-m', '--model_name', type=str, default='large-v2', help='Model to use for speech recognition.')
    parser.add_argument('-n', '--num_speakers', type=int, default=None)

    annote_whisper(**vars(parser.parse_args()))
