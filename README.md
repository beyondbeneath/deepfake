# Deep Fake

Random collection of code snippets used to create Deep Fakes. It is not expected these would be useful to the public, rather it's just a collection of js, Python and bash commands we used throughout the process.

## Repos

* Face Swap: [deepfakes/faceswap](https://github.com/deepfakes/faceswap)
* Voice synthesis alignment (text -> mel spectrograms): [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2/)
* Voice synthesis vocoder (mel spectrograms -> wave): [NVIDIA/waveglow](https://github.com/NVIDIA/waveglow)
* YouTube downloader: [ytdl-org/youtube-dl](https://github.com/ytdl-org/youtube-dl)

## Face Swap

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
