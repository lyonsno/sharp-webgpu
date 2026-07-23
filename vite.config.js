import { defineConfig } from 'vite';
import { resolve } from 'node:path';

const inlineBuild = process.env.SHARP_INLINE_BUILD === '1';

export default defineConfig({
  base: inlineBuild ? '/sharp-inline/' : '/',
  publicDir: inlineBuild ? false : 'public',
  server: {
    port: 5175,
    open: true,
  },
  build: {
    target: 'esnext',
    ...(inlineBuild ? {
      outDir: 'dist-inline',
      emptyOutDir: true,
      lib: {
        entry: resolve(process.cwd(), 'src/main.js'),
        formats: ['es'],
        fileName: 'sharp-inline',
      },
    } : {}),
  },
});
