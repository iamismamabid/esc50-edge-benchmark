import http from 'node:http';
import { pipeline } from '@huggingface/transformers';
import { WaveFile } from 'wavefile';

const dtype = (process.env.MODEL_DTYPE as 'fp32' | 'q8') || 'q8';
const serverStart = performance.now();
let classifier: any = null;
let modelLoadMs = 0;

async function getClassifier() {
  if (!classifier) {
    const t0 = performance.now();
    classifier = await pipeline(
      'audio-classification',
      'Xenova/ast-finetuned-audioset-10-10-0.4593',
      { dtype }
    );
    modelLoadMs = performance.now() - t0;
  }
  return classifier;
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'POST' && req.url === '/classify') {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk);
    const buffer = Buffer.concat(chunks);

    const wav = new WaveFile(buffer);
    wav.toBitDepth('32f');
    wav.toSampleRate(16000);
    const audioData = wav.getSamples(false, Float32Array) as unknown as Float32Array;

    const clf = await getClassifier();
    const t0 = performance.now();
    const result = await clf(audioData, { topk: 1 });
    const inferMs = performance.now() - t0;

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      dtype,
      modelLoadMs: Number(modelLoadMs.toFixed(2)),
      inferMs: Number(inferMs.toFixed(2)),
      rssMb: Number((process.memoryUsage().rss / (1024 * 1024)).toFixed(1)),
      prediction: result[0]
    }));
    return;
  }

  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end(`ESC-50 Cloud Run API ready (uptime ${(performance.now() - serverStart).toFixed(0)} ms)`);
});

const PORT = Number(process.env.PORT) || 8080;
server.listen(PORT, () => console.log(`Listening on port ${PORT} (dtype=${dtype})`));
