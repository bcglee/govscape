import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

/** @type {import('vite').UserConfig} */
export default defineConfig({
	plugins: [sveltekit()],
	server: {
		fs: {
			strict: false
		},
		hmr: {
			overlay: false
		},
		proxy: {
			'/api': {
				target: 'http://localhost:8080',
				changeOrigin: true
			}
		}
	},
	build: {
		sourcemap: true
	},
	optimizeDeps: {
		force: true
	}
});
