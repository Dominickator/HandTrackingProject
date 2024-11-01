import argparse
import queue
import sys
import sounddevice as sd
from pyautogui import write
import json

from vosk import Model, KaldiRecognizer

def result_to_text(jsonResult, wfAccpet):
    tmpData = json.loads(jsonResult)
    if wfAccpet:
        textData = json.dumps(tmpData['text'])
    else:
        textData = json.dumps(tmpData['partial'])
    
    textData = textData.replace('"', '')
    return textData

q = queue.Queue()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l", "--list-devices", action="store_true",
    help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    "-d", "--device", type=int_or_str,
    help="input device (numeric ID or substring)")
parser.add_argument(
    "-s","--samplerate", type=int_or_str
)
args = parser.parse_known_args()

try:
    device_info = sd.query_devices(args[0].device, "input")
    # soundfile expects an int, sounddevice provides a float:
    args[0].samplerate = int(device_info["default_samplerate"])
    model = Model(lang="en-us")

    buffer_queue = queue.Queue()

    with sd.RawInputStream(samplerate=args[0].samplerate, blocksize = 8000, device=args[0].device,
            dtype="int16", channels=1, callback=callback):
        print("#" * 80)
        print("Press Ctrl+C to stop the recording")
        print("#" * 80)

        rec = KaldiRecognizer(model, args[0].samplerate)
        while True:
            data = q.get()
            wfAccepted = rec.AcceptWaveform(data)
            if wfAccepted:
                fullResult = result_to_text(rec.Result(),wfAccepted)
                if fullResult != '':
                    buffer_queue.put(fullResult)
                    print(f"Waveform accepted, recognized word/phrase: {fullResult}")
            else:
                parResult = result_to_text(rec.PartialResult(),wfAccepted)
                if parResult != '':
                    #buffer_queue.put(parResult)
                    print(f"Waveform partially accepted, recognized word/phrase: {parResult}")
            
            while not buffer_queue.empty():
                write(buffer_queue.get())


except KeyboardInterrupt:
    print("\nDone")
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ": " + str(e))
