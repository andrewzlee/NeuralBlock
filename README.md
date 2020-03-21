# NeuralBlock
NeuralBlock (NB) is a neural network built using Keras/Tensorflow that detects in-video YouTube sponsorships. There is support for both predicting (1) whether or not a text excerpt is a sponsorship or (2) whether or not this word in the sequence is part of a sponsorship. Providing more real life utility, development going forward will be focused on (2).

This project is loosely based on and inspired by the project https://github.com/Sponsoff/sponsorship_remover. Unlike the aforementioned project, this project leverages the crowd-sourced labels of https://github.com/ajayyy/SponsorBlock.

Some examples of NB's predictions are provided in the `examples/` directory. A web application can also be run locally for a more hands-on experience. 

## High Level Summary
1. NeuralBlock extract transcripts from youtube videos via https://pypi.org/project/youtube-transcript-api/ python library.
2. The SponsorBlock community has already pre-labeled sponsors.
3. The timestamps from (2) are used to label the transcript word for word, thereby creating a training set.
4. The sequence of text is tokenized using the top 10,000 words found in sponsorships. A word embedding using [fastText](https://fasttext.cc/) is generated.
5. Finally, a bidirectional LSTM RNN is trained. The network outputs, at every word, a probability whether or not this word is part of a sponsorship.

## Using the Web App
The `app/` directory contains a simple flask application that performs the primary functions of `predict_stream.py` and `predict_timestamps.py`, and presents the results in the broswer.

1. Install flask and other necessary libraries.
2. Run `python app.py` from a terminal.
3. Go to `localhost:5000` in a broswer.
4. Submit a valid video ID and click Submit

The results should return in a few seconds. Note, if a good transcript cannot be extracted by YouTubeTranscriptApi, the app will fail.

## Predicting On New Data
1. Install the python libraries tensorflow and YouTubeTranscriptApi
2. Update paths if necessary
3. Provide a video id (vid). The network was trained on the database as of 3/3/20. Use a video that was created after that date to ensure that the video hasn't already been seen.
4. Run predict_stream.py
5. Manually inspect the output stored in the variable `df` or `results`. 

## Future Work
1. **Better transcripts:** some creators do not caption the sponsored segments of their videos. NeuralBlock depends on being able to download the full closed captioning. Further, some creators do not even have auto-generated English captions, making it impossible for NB to predict on. The latter could be resolved through existing text-to-speech projects such as https://github.com/mozilla/DeepSpeech.
2. **Support for other languages:** Only English is supported at this moment.
3. **Incorporate video:** Visual cues, such as scene cuts, are also valuable in determining ads
4. **More accurate labels:** The data may still be dirty. Labeling the start and end is challenging