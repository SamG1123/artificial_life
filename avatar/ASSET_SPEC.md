# Avatar Asset Specification (v1)

This format is for the next step when replacing the placeholder overlay face with your own 2D art.

## Folder layout

```text
avatar_assets/
  base/
    body.png
    head.png
  eyes/
    neutral.png
    happy.png
    angry.png
    blink.png
  brows/
    neutral.png
    raised.png
    frown.png
  mouth/
    rest.png
    aa.png
    ee.png
    ih.png
    oh.png
    uh.png
    mbp.png
    fv.png
    td.png
    sz.png
    kg.png
    l.png
    r.png
    wq.png
  expressions/
    calm.json
    amused.json
    frustrated.json
```

## Image requirements

- Format: PNG with alpha channel.
- Canvas size: 1024x1024 recommended.
- Anchor point: center-based alignment for all layers.
- Coordinate system: same pixel origin for every layer file.

## Expression JSON format

Each expression JSON controls layer visibility and blend strengths.

```json
{
  "name": "amused",
  "eyes": "happy.png",
  "brows": "raised.png",
  "mouth_bias": "ee",
  "strength": 0.8
}
```

## Viseme set used by runtime

The current viseme IDs expected by code are:

- rest
- aa
- ee
- ih
- oh
- uh
- mbp
- fv
- td
- sz
- kg
- l
- r
- wq

## Notes

- Keep mouth sprites tightly cropped to reduce blending artifacts.
- Blink frame should be full eyelid closure.
- If a viseme sprite is missing, runtime falls back to `rest`.
- Overlay position is persisted to `memory_store/avatar_overlay_state.json` after drag.

## VTube Studio Expression Mapping

You can map emotion names to VTube Studio expression files via env var:

```powershell
$env:AVATAR_VTS_EXPRESSION_MAP = '{"amusement":"amused.exp3.json","surprise":"surprised.exp3.json","frustration":"angry.exp3.json"}'
```

Emotion keys used by runtime include:

- amusement
- satisfaction
- pride
- curiosity
- surprise
- frustration
- disappointment
- concern
