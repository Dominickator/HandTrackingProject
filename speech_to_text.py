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

def run_stt(device, stop_event):
    try:
        device_info = sd.query_devices(device, "input")
        # soundfile expects an int, sounddevice provides a float:
        samplerate = int(device_info["default_samplerate"])
        model = Model(lang="en-us")

        buffer_queue = queue.Queue()

        with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=device,
                dtype="int16", channels=1, callback=callback):
            print("#" * 80)
            print("Press Ctrl+C to stop the recording")
            print("#" * 80)

            rec = KaldiRecognizer(model, samplerate)
            while not stop_event.is_set():
                data = q.get()
                wfAccepted = rec.AcceptWaveform(data)
                if wfAccepted:
                    fullResult = result_to_text(rec.Result(),wfAccepted)
                    fullResult = recognizeCommands(fullResult)
                    if fullResult != '':
                        fullResult = formatGrammar(fullResult)
                        buffer_queue.put(fullResult)
                        print(f"Waveform accepted, recognized word/phrase: {fullResult}")
                else:
                    parResult = result_to_text(rec.PartialResult(),wfAccepted)
                    if parResult != '':
                        #buffer_queue.put(parResult)
                        print(f"Partial waveform recognized, recognized word/phrase: {parResult}")
                
                while not buffer_queue.empty():
                    write(buffer_queue.get())

    except KeyboardInterrupt:
        print("\nDone")

def recognizeCommands(input:str):
    commandPhrase = ["period", "new line", "enter", "back space", "delete"]
    input = input.lower()
    if input in commandPhrase:
        if input == "period":
            return ". "
        if input == "new line" or input =="enter":
            return "\n"
        if input == "back space" or input =="delete":
            return "\b"
    else:
        return input



#Currently very rudimentary, Vosk/Kaldi handles some, this applies some other common general cases.
def formatGrammar(input:str):
    #capitalize first letter

    input = input.replace(input[:1], input[:1].upper(), 1)

    #Append period and space to end of waveform, only if longer than 10 characters.
    #Otherwise it is highly likely to either be a command character or intended as a comma
    if len(input) > 10:
        input += ". "
    elif len(input) > 2:
        input += ", "

    #Capitalizing I
    input = input.replace(" i ", " I ")
    input = input.replace(" i'", " I'")

    return input