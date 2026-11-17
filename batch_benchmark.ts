import { pipeline } from '@huggingface/transformers';
import { WaveFile } from 'wavefile';
import fs from 'node:fs';
import path from 'node:path';

function loadAudio16kHz(filePath: string): Float32Array {
  const buffer = fs.readFileSync(filePath);
  const wav = new WaveFile(buffer);
  wav.toBitDepth('32f');
  wav.toSampleRate(16000);
  return wav.getSamples(false, Float32Array) as unknown as Float32Array;
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

// ESC-50-এর ৫০টি ক্লাসের প্রতিটি থেকে ১টি করে মোট ৫০টি ফাইল নেবে
function getSample50Clips(audioDir: string): { file: string; target: string }[] {
  const files = fs.readdirSync(audioDir).filter((f) => f.endsWith('.wav'));
  const seenTargets = new Set<string>();
  const selected: { file: string; target: string }[] = [];

  for (const file of files) {
    const target = file.replace('.wav', '').split('-').pop() || '';
    if (!seenTargets.has(target)) {
      seenTargets.add(target);
      selected.push({ file: path.join(audioDir, file), target });
    }
    if (selected.length === 50) break;
  }
  return selected;
}

async function benchmarkDtype(dtype: 'fp32' | 'q8', clips: { file: string; target: string }[]) {
  console.log(`\n=== Running Local Node.js Benchmark (${dtype}) ===`);
  const loadStart = performance.now();
  const classifier = await pipeline(
    'audio-classification',
    'Xenova/ast-finetuned-audioset-10-10-0.4593',
    { dtype }
  );
  const loadMs = performance.now() - loadStart;

  // Warm-up
  const firstAudio = loadAudio16kHz(clips[0].file);
  await classifier(firstAudio, { topk: 1 });

  const latencies: number[] = [];
  const rows: string[] = [];

  for (let i = 0; i < clips.length; i++) {
    const audio = loadAudio16kHz(clips[i].file);
    const t0 = performance.now();
    const out = (await classifier(audio, { topk: 1 })) as { label: string; score: number }[];
    const t1 = performance.now();
    const ms = t1 - t0;
    latencies.push(ms);
    rows.push(`local_node,${dtype},${path.basename(clips[i].file)},${clips[i].target},"${out[0].label}",${out[0].score.toFixed(4)},${ms.toFixed(2)}`);
    if ((i + 1) % 10 === 0) console.log(`Processed ${i + 1}/50 clips...`);
  }

  const sorted = [...latencies].sort((a, b) => a - b);
  const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const p50 = percentile(sorted, 50);
  const p95 = percentile(sorted, 95);
  const rssMb = process.memoryUsage().rss / (1024 * 1024);

  console.log(`[${dtype}] Cold-Start: ${loadMs.toFixed(2)} ms | Avg: ${avg.toFixed(2)} ms | p50: ${p50.toFixed(2)} ms | p95: ${p95.toFixed(2)} ms | RAM (RSS): ${rssMb.toFixed(1)} MB`);
  return rows;
}

async function main() {
  const clips = getSample50Clips('./audio');
  console.log(`Found ${clips.length} unique class clips.`);

  const header = 'env,dtype,file,esc50_target,top_label,confidence,latency_ms';
  const fp32Rows = await benchmarkDtype('fp32', clips);
  const q8Rows = await benchmarkDtype('q8', clips);

  fs.writeFileSync('local_results.csv', [header, ...fp32Rows, ...q8Rows].join('\n'), 'utf8');
  console.log('\nSaved all results to local_results.csv!');
}

main().catch(console.error);
