#Find a way to download audio in 16khz
#youtube-dl?

#Rebuild client.py to accept downloaded video

#return ordered dictionary 
#also string form so we don't have to rebuild from dictionary


#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json

from deepspeech import Model
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

from youtube_dl import YoutubeDL

def yt_download(vid):
    ydl_opts = {
        'format': 'worstaudio',
        'outtmpl': '%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        #'postprocessor_args': [
        #    '-ar', '16000'
        #]
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([vid])

    return

def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)

def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time "] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list

def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)

def metadata_json_output(transcript):
    json_result = dict()
    json_result["transcript"] = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    }]
    return json.dumps(json_result, indent=2)



def transcribe(audio_path):
    ds = Model(model_path = "deepspeech-0.7.0-models.pbmm")

    desired_sample_rate = ds.sampleRate()
    print(desired_sample_rate)
    ds.enableExternalScorer("deepspeech-0.7.0-models.scorer")

    fin = wave.open(audio_path, 'rb')
    fs_orig = fin.getframerate()
    
    if fs_orig != desired_sample_rate:
        print("Converting from {}hz to {}hz" % (fs_orig, desired_sample_rate))
        fs_new, audio = convert_samplerate(audio_path, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()

    inference_start = timer()
    transcript = ds.sttWithMetadata(audio, 1).transcripts[0]
    json_result = metadata_json_output(transcript)
    string_result = metadata_to_string(transcript)

    inference_end = timer() - inference_start
    print(json_result)
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
    
    return json_result, string_result

if __name__ == '__main__':
    #yt_download("PCuJJ6thps4")
    transcribe("PCuJJ6thps4.wav")
