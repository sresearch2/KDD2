import os

import librosa
import pandas as pd
import torch
from pytube import YouTube


def download_youtube_video(url, custom_filename):
    """
    Download a YouTube video in 360p resolution.

    Args:
        url (str): The URL of the YouTube video.
        custom_filename (str): The desired filename for the downloaded video.

    Returns:
        str: The path to the downloaded video.
    """
    yt = YouTube(url)
    streams = yt.streams.filter(res="360p")

    if not streams:
        raise ValueError(f"No '360p' resolution available for the video.")
    if not os.path.exists("videos"):
        os.mkdir("videos")
    stream = streams.first()
    video_path = stream.download(output_path="videos", filename=custom_filename)
    return video_path


class SpeakerDiarization:
    """
    Class for speaker diarization of audio in a video.
    """

    def __init__(self, model, pipeline):
        """
        Initialize the SpeakerDiarization object.

        Args:
            model: The model for speech recognition.
            pipeline: The diarization pipeline.
        """
        self.model = model
        self.pipeline = pipeline

    def get_speaker_separated_script(self, df, speaker_timings):
        """
        Generate a script with speaker-separated text based on speaker timings.

        Args:
            df (pandas.DataFrame): DataFrame containing word-level timestamps.
            speaker_timings (list): List of speaker timings.

        Returns:
            str: Speaker-separated script.
        """
        script = ""
        past_speaker = ""
        for start, end, speaker in speaker_timings:
            if past_speaker != speaker:
                script += f"\n{speaker}: "
            past_speaker = speaker
            script += " ".join(
                [
                    x.strip()
                    for x in df[df["start"] > start][df["end"] < end].word.values
                ]
            )
        return script

    def get_script(self, video):
        """
        Extract a speaker-separated script from the given video.

        Args:
            video (str): Path to the video file.

        Returns:
            str: Speaker-separated script.
        """
        audio = librosa.load(video, sr=16000)
        audio_in_memory = {
            "waveform": torch.tensor(audio[0]).unsqueeze(0),
            "sample_rate": 16000,
        }
        diarization = self.pipeline(audio_in_memory)
        speaker_timings = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timings.append([turn.start, turn.end, speaker])

        audio_in_memory = {
            "waveform": torch.tensor(audio[0]).unsqueeze(0),
            "sample_rate": 16000,
        }
        transcript = self.model.transcribe(word_timestamps=True, audio=audio[0])
        diarization = self.pipeline(audio_in_memory)
        speaker_timings = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timings.append([turn.start, turn.end, speaker])
        df = pd.DataFrame([word for x in transcript["segments"] for word in x["words"]])
        script = self.get_speaker_separated_script(df, speaker_timings)
        return script
