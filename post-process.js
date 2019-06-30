const data = require('./data.json'); // Results speech-api.js
const fs = require('fs');
const { spawn } = require('child_process');

const SOURCE_FILES_PATH = '<WHERE ORIGINAL FILES ARE>';
const RESULT_SHORT_WAVS_PATH = `${SOURCE_FILES_PATH}/short WAVs`;
const RESULT_LONG_PAUSES_WAVS_PATH = `${SOURCE_FILES_PATH}/long pauses WAVs`;
const RESULT_LONG_WAVS_PATH = `${SOURCE_FILES_PATH}/long WAVs`;

const processData = (fileData = data[0]) => {
  const { filename } = fileData;
  const words = fileData.response.results.reduce(((arr, res) => arr.concat(res.alternatives[0].words)), []);
  const wordsInSentences = [[]];
  words.filter(el => el).forEach(el => {
    let { startTime, endTime, word } = el;

    wordsInSentences[wordsInSentences.length - 1].push({ startTime, endTime, word });
    if (word.endsWith('.')) {
      wordsInSentences.push([]);
    }
  });

  wordsInSentences.splice(-1, 1);
  const parseTime = (time) => parseFloat(time.substr(0, time.length - 1));

  const parseTimeToMinSecFormat = time => {
    const totalSeconds = parseTime(time);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toFixed(2)}`;
  };

  const sentences = wordsInSentences.map((words, i) => {
    let pauseAccum = 0;
    let longestPause = -Number.MAX_VALUE;
    for (let i = 0; i < words.length - 1; i++) {
      const pause = parseTime(words[i + 1].startTime) - parseTime(words[i].endTime);
      pauseAccum += pause;
      if (pause > longestPause) {
        longestPause = pause;
      }
    }
    const averagePause = pauseAccum / (words.length - 1);

    const startTime = words[0].startTime;
    const endTime = words[words.length - 1].endTime;
    const lenInSec = parseTime(endTime) - parseTime(startTime);
    const skip = parseTimeToMinSecFormat(startTime);
    const until = parseTimeToMinSecFormat(endTime);
    return {
      lenInSec,
      longestPause: longestPause || 0,
      averagePause: averagePause && averagePause > 0 && averagePause || 0,
      skip,
      until,
      transcript: words.map(({ word }) => word).join(' ')
    }
  });

  return { filename, sentences };
};

const cutToWav = async ({ inputFile, outFile, skip, until }) => new Promise((resolve, reject) => {
  const ls = spawn('flac', [
    `--skip=${skip}`,
    `--until=${until}`,
    '-d',
    '-f',
    `--output-name=${outFile}`,
    inputFile]);

  let error = '';
  ls.stderr.on('data', (data) => {
    error += data;
  });

  ls.on('close', (code) => {
    if (code === 0) {
      resolve();
    } else {
      reject(error);
    }
  });
});

const main = async (maxSecs = 360) => {
  const meta = [];
  for (let i = 0; i < data.length; i++) {
    let { filename, sentences } = processData(data[i]);
    sentences = sentences.sort((a, b) => a.longestPause - b.longestPause);
    const baseName = filename.split('.')[0].toUpperCase();
    const inputFile = `${SOURCE_FILES_PATH}/${filename}`;
    console.log(`Starting ${filename}`);
    let lastPercentPrinted = undefined;
    for (let j = 0; j < sentences.length; j++) {
      const percent = Math.round(j / sentences.length * 100);
      if (percent % 10 === 0 && lastPercentPrinted !== percent) {
        console.log(`${percent}%`);
        lastPercentPrinted = percent;
      }

      const { skip, until, transcript, lenInSec, longestPause } = sentences[j];
      if (lenInSec <= maxSecs && longestPause < 1) {
        const outFile = `${RESULT_SHORT_WAVS_PATH}/${baseName}-${j}.WAV`;
        await cutToWav({ inputFile, outFile, skip, until });
        meta.push(`DUMMY/${baseName}-${j}.WAV|${transcript}`);
      } else if (longestPause >= 1) {
        const outFile = `${RESULT_LONG_PAUSES_WAVS_PATH}/${baseName}-${j}.WAV`;
        await cutToWav({ inputFile, outFile, skip, until });
      } else {
        const outFile = `${RESULT_LONG_WAVS_PATH}/${baseName}-${j}.WAV`;
        await cutToWav({ inputFile, outFile, skip, until });
      }
    }
  }
  fs.writeFileSync(`${RESULT_SHORT_WAVS_PATH}/short-meta-with.txt`, meta.join('\n'));
  console.log('done');
};

main(10);




