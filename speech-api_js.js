const fetch = require('node-fetch');
const util = require('util');
const fs = require('fs');
const KEY = '<GOOGLE API KEY>';
const GOOGLE_STORAGE_BUCKET = 'gs://<BUCKET-PATH>';
const FILES_IN_BUCKET = [
  '<FILES-IN-BUCKET>'
];

const getOperations = async (name) => {
  const response = await fetch(`https://speech.googleapis.com/v1/operations/${name}?key=${KEY}`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
};

const startProcessing = async (filename) => {
  const url = `https://speech.googleapis.com/v1/speech:longrunningrecognize?key=${KEY}`;
  const uri = `${GOOGLE_STORAGE_BUCKET}/${filename}`;
  const response = await fetch(url, {
    method: 'POST',
    body: JSON.stringify({
      config: {
        enableWordTimeOffsets: true,
        enableAutomaticPunctuation: true,
        useEnhanced: true,
        audioChannelCount: 1,
        model: 'video',
        languageCode: 'en-US'
      },
      "audio": {
        uri
      }
    })
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
};

const processFile = async (filename) => {
  const { name } = await startProcessing(filename);
  let done = false;
  let operation;
  while (!done) {
    operation = await getOperations(name);
    done = operation.done || false;
    const progress = operation.metadata.progressPercent;
    console.log(`${filename} - ${progress}% waiting ....`);
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  console.log(util.inspect(operation, { depth: null, colors: true }));
  return { ...operation, filename };
};

const main = async () => {
  const promises = FILES_IN_BUCKET.map(processFile);
  const operations = await Promise.all(promises);
  fs.writeFileSync('data.json', JSON.stringify(operations))
};

main();
