import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';

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

function gitOutput(args, cwd = process.cwd()) {
  try {
    return execFileSync('git', args, {
      cwd,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return '';
  }
}

const witnessSourceIdentity = {
  name: 'sharp-witness-source-identity',
  apply: 'serve',
  configureServer(server) {
    const servedRoot = resolve(server.config.root);
    const gitRoot = resolve(gitOutput(['rev-parse', '--show-toplevel'], servedRoot) || servedRoot);
    const servedSourceRevision = gitOutput(['rev-parse', 'HEAD'], servedRoot);
    const servedSourceState = servedRoot === gitRoot
      && gitOutput(['status', '--porcelain'], servedRoot) === ''
      ? 'clean'
      : servedRoot === gitRoot ? 'dirty' : 'root-mismatch';
    server.middlewares.use((request, response, next) => {
      response.setHeader('x-sharp-source-revision', servedSourceRevision);
      response.setHeader('x-sharp-source-state', servedSourceState);
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
