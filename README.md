# NeuralBlock
NeuralBlock (NB) is a neural network built using Keras/Tensorflow that detects in-video YouTube sponsorships. There is support for both predicting [1] whether or not a text excerpt is a sponsorship (spot) or [2] whether or not this word in the sequence is part of a sponsorship.

NB is loosely based on and inspired by this [project](https://github.com/Sponsoff/sponsorship_remover). Unlike the aforementioned project, this project leverages the crowd-sourced labels provided by [SponsorBlock](https://github.com/ajayyy/SponsorBlock).

Some examples of NB's predictions are provided in the `examples/` directory. The code for the (web application)[ai.neuralblock.app] is also provided. can also be run locally for a more hands-on experience.

## High Level Summary
1. NeuralBlock extracts transcripts from YouTube with (YouTubeTranscriptApi)[https://pypi.org/project/youtube-transcript-api/].
2. The SponsorBlock community has already pre-labeled sponsors.
3. The timestamps from (2) are used to find the sections in the transcript that are sponsorships, thereby creating a training set.
4. The sequence of text is tokenized using the top 10,000 words found in sponsorships. Note, using a pre-trained word embedding by [fastText](https://fasttext.cc/) does not yield better performance.
5. A bidirectional LSTM RNN is trained.

## Using the Web App
**Somewhat outdated. To be updated later. Dockerfile can be used**
The `app/` directory contains a simple flask application that performs the primary functions of `predict_stream.py` and `predict_timestamps.py`, and presents the results in the browser.

1. Install flask and other necessary libraries.
2. Move the models from the `data` folder into `app/models`. There should be no subfolders.
3. Run `python app/application.py` from a terminal.
4. Go to `localhost:5000` in a browser.
5. Submit a valid video ID and click Submit

The results should return in a few seconds. Note, if a good transcript cannot be extracted by YouTubeTranscriptApi, the app will fail.

## Predicting On New Data
**Somewhat outdated. To be updated later.**
1. Install the python libraries TensorFlow and YouTubeTranscriptApi
2. Update paths if necessary
3. Provide a video id (vid). The network was trained on the database as of 3/3/20. Use a video that was created after that date to ensure that the video hasn't already been seen.
4. Run predict_stream.py
5. Manually inspect the output stored in the variable `df` or `results`.

## Future Work
1. **Better transcripts:** NeuralBlock depends on being able to download the full closed captioning. Some creators disallow auto-generated English captions, making it impossible for NB to predict on. The latter could be resolved through existing speech-to-text projects such as Mozilla's [DeepSpeech](https://github.com/mozilla/DeepSpeech).
2. **More accurate labels:** The labels is imperfect because we don't know the moment a word is spoken, only an approximate time. For example, silence (visual only ad) or really short ad segments are hard to account for.
3. **Incorporate video:** Visual cues, such as scene cuts, are also valuable in determining ads and can help with (2).
4. **Support for other languages:** Only English is supported at this moment.
