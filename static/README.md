# India XI — player identification frontend

A responsive, light Indian-cricket-themed vanilla HTML/CSS/JS frontend. Open `index.html` to preview it. No build step or external JavaScript dependencies are required. Google Fonts are loaded from Google Fonts; system sans-serif fallbacks are provided.

## Files

- `index.html` — page structure and accessible upload/result states
- `styles.css` — responsive styling, CSS-drawn subtle stadium, and scan animation
- `app.js` — drag/drop, JPG/PNG/WEBP validation, preview, replace/remove controls, UI state, and integration bridge

## Connect the existing recognition API

The page does not invent or call an API URL. It dispatches `indianxi:identify` when the user presses **IDENTIFY PLAYER**. Add your project-specific integration code after `app.js` (or in your app bundle):

```js
window.addEventListener('indianxi:identify', async (event) => {
  const { file, previewUrl } = event.detail;
  const formData = new FormData();
  formData.append('image', file); // change field name to match your existing backend

  try {
    // Replace this with your existing endpoint and request/response handling.
    const response = await fetch('/YOUR-EXISTING-ENDPOINT', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error('Recognition request failed');
    const data = await response.json();

    // Map your real response fields into these UI fields.
    window.IndiaXI.showResult({
      name: data.name,
      confidence: data.confidence, // accepts 0–1 or 0–100
      detail: data.detail || 'India · International cricketer',
      imageUrl: previewUrl
    });
  } catch (error) {
    window.IndiaXI.showError(error.message || 'Please try again in a moment.');
  }
});
```

For a low-confidence or unknown-person response, call `window.IndiaXI.showUnknown('Optional user-facing message')`. The default confidence threshold is `0.50`; set another 0–1 threshold with `window.IndiaXI.setConfidenceThreshold(0.65)`. `window.IndiaXI.setLoading()`, `.showResult(...)`, `.showUnknown(...)`, and `.showError(...)` are available to your integration code.

Keep your current backend endpoint, request field names, authentication, and response mapping: the example above is a connection template, not a replacement API.
