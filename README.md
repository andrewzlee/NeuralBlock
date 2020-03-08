# NeuralBlock
NeuralBlock is a neural network built using Keras/Tensorflow that detects in-video YouTube sponsorships. There is support for both predicting (1) whether or not a text excerpt is a sponsorship or (2) whether or not this word in the sequence is part of a sponsorship.

This project is loosely based on and inspired by the project https://github.com/Sponsoff/sponsorship_remover. Unlike the aforementioned project, this project leverages the crowd-sourced labels of https://github.com/ajayyy/SponsorBlock.

# High Level Summary
1. NeuralBlock extract transcripts from youtube videos via https://pypi.org/project/youtube-transcript-api/ python library.
2. The SponsorBlock community has already pre-labeled sponsors.
3. The timestamps from (2) are used to label the transcript word for word, thereby creating a training set.
4. The sequence of text is tokenized using the top 10,000 words found in sponsorships. A word embedding using fastText is generated.
5. Finally, a bidirectional LSTM is trained. The network outputs, at every word, a probability whether or not this word is part of a sponsorship.
