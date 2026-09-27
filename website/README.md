# WIUT Traffic Vision website

The Vite frontend includes the project narrative, held-out metrics, and a live
video upload interface backed by the local GPU API in `demo_api.py`.

For the complete Windows setup and launch commands, see
[`../LIVE_DEMO.md`](../LIVE_DEMO.md).

## Frontend development

Run the API on port 8000, then:

```bash
npm install
npm run dev
```

Set `VITE_API_URL=http://127.0.0.1:8000` when the API is on a different origin.

## Vite notes

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
