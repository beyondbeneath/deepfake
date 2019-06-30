# Deep Fake

Random collection of code snippets used to create Deep Fakes using Databricks for the GPU computation components. It is not expected these would be useful to the public, rather it's just a collection of js, Python and bash commands we used throughout the process.

## Repos

* Face Swap: [deepfakes/faceswap](https://github.com/deepfakes/faceswap)
* Voice synthesis alignment (text -> mel spectrograms): [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2/)
* Voice synthesis vocoder (mel spectrograms -> wave): [NVIDIA/waveglow](https://github.com/NVIDIA/waveglow)
* YouTube downloader: [ytdl-org/youtube-dl](https://github.com/ytdl-org/youtube-dl)

## Face Swap

### Environment configuration

Prepare Databricks environment:

```bash
cd /dbfs/.../faceswap
sudo /databricks/python3/bin/pip install --upgrade pip
sudo /databricks/python3/bin/pip install -r requirements.txt
sudo apt-get install ffmpeg
```

### Training data collection

Download YouTube videos from text file:

```bash
youtube-dl \
--batch-file "data/url_list.txt" \
-o 'data/videos/%(autonumber)s.%(ext)s' \
--autonumber-start 1 \
```

Trim videos to appropriate sections:

```bash
ffmpeg -ss 0:10 -to 1:10 -i 00001.mp4 -codec copy 00001a.mp4
```

Extract every 12th frame from a video (see [StackOverflow post](https://stackoverflow.com/questions/35912335/how-to-extract-a-fixed-number-of-frames-with-ffmpeg)):

```bash
ffmpeg -i videos00001a.mp4 \
-vf select='not(mod(n\,12))',setpts=N/TB \
-r 1 video-frames/video1a-%04d.jpg
```

Extract faces from directory of frames:

```bash
python faceswap.py extract -i ../data/video-frames -o ../data/faces
```

### Model build

Build a model on a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 /databricks/python3/bin/python faceswap.py train \
-A /dbfs/.../facesA \
-B /dbfs/.../facesB \
-m /dbfs/.../model-output \
--timelapse-input-A /dbfs/.../facesA-tl \
--timelapse-input-B /dbfs/.../facesB-tl \
--timelapse-output /dbfs/.../model-tl \
--write_image \
-t dfaker \
--batch-size 16
```

Note that you should put some representative faces of person A and person B (e.g, 6 of each) in the `facesA-tl` and `facesB-tl` folders, which will produce a frame by frame animation as your model builds. Once the model is built, you can convert the frames into a nice animation at (e.g.,) 10 frames-per-second:

```bash
ffmpeg -framerate 10 -pattern_type glob -i '*.jpg' model-tl.mp4
```

### Inference

Extract faces from a video file, typically in preparation for inference:

```bash
python faceswap.py extract -i ../data/videos/eval.mp4 -o ../data/video-frames-eval/
```

Remove the all but the first face detection from each frame, using the output `alignments.json` from previoius step (warning: need to verify that in each case the first detection is indeed the one you care about):

```js
const data = require('./alignments.json');
const fs = require('fs');
let finalObject = {};
Object.entries(data).forEach(([key, value]) => {
  finalObject[key] = [value[0]];
});
fs.writeFileSync('alignments-filtered.json', JSON.stringify(finalObject));
```

Make the actual face swap on your video using a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=` /databricks/python3/bin/python faceswap.py convert \
-i /dbfs/..data/eval.mp4 \
-al /dbfs/../data/alignments-filtered.json \
-o /dbfs/../converted \
-m /dbfs/../model-output \
-w ffmpeg
```

## Voice synthesis

### Environment configuration

Prepare Databricks environment:

```bash
sudo /databricks/python3/bin/pip install torch==1.0
cd /dbfs/.../tacotron2
sudo /databricks/python3/bin/pip install --upgrade pip
sudo /databricks/python3/bin/pip install -r requirements.txt
```

### Training data collection

Use the Google Speech API to transcript a bunch of videos:

* [speech-api.js]()

Convert the transcribed videos into a bunch of smaller `.wav` files and create a metadata `.txt` which can be used as the labelled set in Tacotron2:

* [post-processing.js]()

Convert the files to the correct bitrate:

```bash
for f in *.WAV; do ffmpeg -i "$f" -acodec pcm_s16le "16bit/$f"; done
```

### Model build

#### Tacotron (text -> mel-spectrogram)

Fine-tune (transfer learn) the Tacotron model from the NVIDIA [supplied checkpoint](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing):

```bash
CUDA_VISIBLE_DEVICES=1 /databricks/python3/bin/python train.py \
--output_directory ../taco_out \
--log_directory ../taco_log \
--checkpoint_path /dbfs/.../tacotron2/tacotron2_statedict.pt \
--warm_start \
--hparams batch_size=8
```

#### Waveglow (vocoder: mel-spectrogram -> wave)

Adjust the config.json:

```js
checkpoint_path=/dbfs/../tactron2/waveglow/waveglow_256channels.pt
channels=256
```

Fine-tune (transfer learn) the Waveglow model from NVIDIA [supplied checkpoint](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing)

```bash
CUDA_VISIBLE_DEVICES=2 /databricks/python3/bin/python train.py -c config.json
```

### Audio inference

This Python function is adpated from the `inference.ipynb` contained in the NVIDIA Tacotron2 repo, and expanded to include the actual `.wav` generation (for some reason this was never supplied) and also parameterise which Tacotron & Waveglow models to use. By default it uses the supplied checkpoints, so to use the fine-tuned models substitute in the appropriate checkpoints:

```python

import sys
import os
import numpy as np
import torch

sys.path.append('/dbfs/../tacotron2/')
sys.path.append('/dbfs/../tacotron2/waveglow')
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
#from denoiser import Denoiser
import librosa

def text_to_wav(checkpoint_path='/dbfs/../tacotron2_statedict.pt',
                waveglow_path='/dbfs/../waveglow_256channels.pt',
                output_file='output.wav',
               text="This is a Deep Fake voice."):

  hparams = create_hparams()
  hparams.sampling_rate = 22050

  model = load_model(hparams)
  model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
  _ = model.cuda().eval().half()

  waveglow = torch.load(waveglow_path)['model']
  waveglow.cuda().eval().half()
  for k in waveglow.convinv:
      k.float()
  #denoiser = Denoiser(waveglow)

  sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
  sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
  mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
  with torch.no_grad():
      audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

  wav = audio[0].data.cpu().numpy()
  librosa.output.write_wav(os.path.join('/dbfs/.../',output_file), wav.astype(np.float32), hparams.sampling_rate)
```

Notes:

* We found ending input text with a period `.` is important otherwise the model outputs a bunch of stuttering garbage for a few seconds at the end of the speech.
* We couldn't get the "de-noiser" to work, so this is commented out
* Interestingly, we got our best results with a Tacotron model fine-tuned on 10k iterations (with batch size of 8 and the default learning rate) and the default Waveglow vocoder. Using any more Tacotron iterations, or indeed any fine-tuning on the Vocoder made things worse.
