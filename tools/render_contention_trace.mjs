#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

import { renderSharpContentionTraceSvg } from './contention_witness_report.mjs';

function valueAfter(args, name) {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : null;
}

const args = process.argv.slice(2);
const reportPath = valueAfter(args, '--report');
const outputPath = valueAfter(args, '--out');
if (!reportPath || !outputPath) {
  console.error('usage: node tools/render_contention_trace.mjs --report <report.json> --out <trace.svg>');
  process.exit(2);
}

try {
  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  const svg = renderSharpContentionTraceSvg(report);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${svg}\n`);
  console.log(JSON.stringify({ ok: true, report: reportPath, output: outputPath, runId: report.runId }));
} catch (error) {
  console.error(JSON.stringify({
    ok: false,
    report: reportPath,
    output: outputPath,
    failurePhase: 'rendering-contention-trace',
    error: error?.message || String(error),
  }));
  process.exit(1);
}
