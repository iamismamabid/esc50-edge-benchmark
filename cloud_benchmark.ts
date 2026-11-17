import fs from 'node:fs';
import path from 'node:path';

const SERVICE_URL = 'https://esc50-api-707249364970.asia-southeast1.run.app/classify';

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

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

async function main() {
  const clips = getSample50Clips('./audio');
  console.log(`Sending ${clips.length} clips to Cloud Run (${SERVICE_URL})...`);

  const e2eLatencies: number[] = [];
  const inferLatencies: number[] = [];
  const rows: string[] = ['env,dtype,file,esc50_target,top_label,confidence,e2e_ms,server_infer_ms,model_load_ms,rss_mb'];
  let coldStartE2E = 0;
  let coldStartModelLoad = 0;
  let lastRss = 0;

  for (let i = 0; i < clips.length; i++) {
    const audioBuffer = fs.readFileSync(clips[i].file);
    const t0 = performance.now();
    const res = await fetch(SERVICE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: audioBuffer,
    });
    const data = (await res.json()) as any;
    const e2eMs = performance.now() - t0;

    if (i === 0) {
      coldStartE2E = e2eMs;
      coldStartModelLoad = data.modelLoadMs;
      console.log(`Cold-Start Request #1 -> E2E: ${e2eMs.toFixed(2)} ms | Model Load: ${data.modelLoadMs} ms | Infer: ${data.inferMs} ms`);
    } else {
      e2eLatencies.push(e2eMs);
      inferLatencies.push(data.inferMs);
    }

    lastRss = data.rssMb;
    rows.push(`cloud_run,${data.dtype},${path.basename(clips[i].file)},${clips[i].target},"${data.prediction.label}",${data.prediction.score.toFixed(4)},${e2eMs.toFixed(2)},${data.inferMs},${data.modelLoadMs},${data.rssMb}`);

    if ((i + 1) % 10 === 0) console.log(`Processed ${i + 1}/50 clips...`);
  }

  const sortedE2E = [...e2eLatencies].sort((a, b) => a - b);
  const sortedInfer = [...inferLatencies].sort((a, b) => a - b);
  const avgE2E = e2eLatencies.reduce((a, b) => a + b, 0) / e2eLatencies.length;
  const avgInfer = inferLatencies.reduce((a, b) => a + b, 0) / inferLatencies.length;

  console.log('\n=== Cloud Run (q8, 2 vCPU, 2 GiB) Summary ===');
  console.log(`Cold-Start E2E: ${coldStartE2E.toFixed(2)} ms (Model Load: ${coldStartModelLoad} ms)`);
  console.log(`Warm Server Infer -> Avg: ${avgInfer.toFixed(2)} ms | p50: ${percentile(sortedInfer, 50).toFixed(2)} ms | p95: ${percentile(sortedInfer, 95).toFixed(2)} ms`);
  console.log(`Warm End-to-End   -> Avg: ${avgE2E.toFixed(2)} ms | p50: ${percentile(sortedE2E, 50).toFixed(2)} ms | p95: ${percentile(sortedE2E, 95).toFixed(2)} ms`);
  console.log(`Container RAM (RSS): ${lastRss} MB`);

  fs.writeFileSync('cloud_results.csv', rows.join('\n'), 'utf8');
  console.log('\nSaved to cloud_results.csv!');
}

main().catch(console.error);
