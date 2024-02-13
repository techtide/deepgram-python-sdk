import threading
import logging
import sounddevice as sd
import numpy as np

from .errors import DeepgramMicrophoneError
from .constants import LOGGING, CHANNELS, RATE, CHUNK

class Microphone:
    """
    This implements a microphone for local audio input. This uses sounddevice under the hood.
    """

    def __init__(
        self,
        push_callback,
        device_name=None,
        verbose=LOGGING,
        rate=RATE,
        chunk=CHUNK,
        channels=CHANNELS
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(verbose)
        self.exit = threading.Event()

        self.chunk = chunk
        self.rate = rate
        self.channels = channels
        self.device_name = device_name
        self.push_callback = push_callback

        self.stream = None

    def is_active(self):
        """
        returns True if the stream is active, False otherwise
        """
        return self.stream is not None and self.stream.active

    def _callback(self, indata, frames, time, status):
        """
        The callback used to process data in callback mode.
        """
        if self.exit.is_set():
            return

        if status:
            self.logger.warning('Stream overflow: %i', status)

        try:
            self.push_callback(indata)
        except Exception as e:
            self.logger.error("Error while sending: %s", str(e))
            raise

    def start(self):
        """
        starts the microphone stream
        """
        if self.stream is not None:
            raise DeepgramMicrophoneError("Microphone already started")

        self.logger.info("rate: %d", self.rate)
        self.logger.info("chunk: %d", self.chunk)
        self.logger.info("channels: %d", self.channels)
        self.logger.info("device_name: %s", self.device_name)

        self.stream = sd.InputStream(
            samplerate=self.rate,
            blocksize=self.chunk,
            channels=self.channels,
            device=self.device_name,
            callback=self._callback
        )

        self.exit.clear()
        self.stream.start()

        self.logger.notice("start succeeded")

    def finish(self):
        """
        Stops the microphone stream
        """
        self.exit.set()

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.logger.notice("finish succeeded")
