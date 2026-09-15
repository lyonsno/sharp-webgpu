import fs from 'node:fs';
import path from 'node:path';

export function createParityReport({ runId, reference = null }) {
  return {
    model: 'SHARP', runId, status: 'running', reference,
    stages: {}, ply: {}, summary: {},
    failurePhase: 'setup', lastCompletedStage: null, error: null,
  };
}

export function finishParityReport(report, comparisons, numVertices) {
  report.ply = comparisons;
  report.status = 'completed';
  report.failurePhase = null;
  report.summary = {
    numVertices,
    comparedStages: Object.values(report.stages).filter(stage => stage.status === 'compared').length,
    missingStages: Object.entries(report.stages).filter(([, stage]) => stage.status !== 'compared').map(([name]) => name),
  };
}

export function writeParityReport(outputPath, report) {
  fs.mkdirSync(path.dirname(path.resolve(outputPath)), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
}
