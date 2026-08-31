import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { realpathSync } from 'node:fs';
import { resolve } from 'node:path';

import { classifyWitnessSourceIdentity } from './tools/decoder_kernel_witness_contract.mjs';

const inlineBuild = process.env.SHARP_INLINE_BUILD === '1';
const inlineBuildOptions = inlineBuild ? {
  outDir: 'dist-inline',
  emptyOutDir: true,
  lib: {
    entry: resolve(process.cwd(), 'src/main.js'),
    formats: ['es'],
    fileName: 'sharp-inline',
  },
} : {};
const witnessEntryPoint = 'sharp-webgpu-root-v1';

function runGit(args, cwd = process.cwd()) {
  try {
    return {
      ok: true,
      output: execFileSync('git', args, {
      cwd,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      }).trim(),
    };
  } catch (error) {
    return { ok: false, output: '', error: error?.message || String(error) };
  }
}

function canonicalRoot(root) {
  try {
    return realpathSync(root);
  } catch {
    return resolve(root);
  }
}

const witnessSourceIdentity = {
  name: 'sharp-witness-source-identity',
  apply: 'serve',
  configureServer(server) {
    const servedRoot = canonicalRoot(server.config.root);
    const rootResult = runGit(['rev-parse', '--show-toplevel'], servedRoot);
    if (rootResult.ok) rootResult.output = canonicalRoot(rootResult.output);
    const servedIdentity = classifyWitnessSourceIdentity({
      expectedRoot: servedRoot,
      rootResult,
      revisionResult: runGit(['rev-parse', 'HEAD'], servedRoot),
      statusResult: runGit(['status', '--porcelain'], servedRoot),
    });
    server.middlewares.use((request, response, next) => {
      response.setHeader('x-sharp-source-revision', servedIdentity.sourceRevision || 'unverifiable');
      response.setHeader('x-sharp-source-state', servedIdentity.sourceState);
      response.setHeader('x-sharp-source-root', servedIdentity.sourceRoot || servedRoot);
      const pathname = new URL(request.url || '/', 'http://127.0.0.1').pathname;
      if (pathname === '/') {
        response.setHeader('x-sharp-entrypoint', witnessEntryPoint);
      }
      next();
    });
  },
};

export default defineConfig({
  plugins: [witnessSourceIdentity],
  base: inlineBuild ? '/sharp-inline/' : '/',
  publicDir: inlineBuild ? false : 'public',
  server: {
    port: 5175,
    open: true,
  },
  build: {
    target: 'esnext',
    ...inlineBuildOptions,
  },
});
