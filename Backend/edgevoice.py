import asyncio

import edge_tts


def edge_tts_save(text: str, voice: str, filename: str) -> None:
    """
    Synthesize `text` with a Microsoft Edge neural voice (e.g. "en-US-GuyNeural")
    and save it as an MP3.

    Args:
        text (str): The text to speak.
        voice (str): Edge voice short name.
        filename (str): Output MP3 path.
    """
    asyncio.run(edge_tts.Communicate(text, voice).save(filename))
