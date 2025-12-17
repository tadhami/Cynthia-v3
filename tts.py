from noahs_tts import TTS  # if you saved the class in tts.py
tts = TTS()

def example_generator():
    for s in ["hello there", "this is a stream", "last chunk here"]:
        yield s

tts.say_stream(example_generator())