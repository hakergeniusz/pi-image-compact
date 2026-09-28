# pi-image-compact

Turns images into text for models that cannot accept them, so a screenshot is
usable instead of silently disappearing.

## The problem

Give a text-only model an image and pi does not fail — it drops the picture.
In `transformMessages`, before the request is built:

```js
function downgradeUnsupportedImages(messages, model) {
  return model.input.includes("image") ? messages
    : messages.map(msg => /* replace every image block with "(tool image omitted: model n)" */);
}
```

The read tool also adds `[Current model does not support images. The image will
be omitted from this request.]`, and the model is left with a filename and
nothing to look at. `opencode/nemotron-3-ultra-free` answers such a request with
"I cannot read or extract text from the PNG file".

## What this does

Any image bound for a model that cannot accept one is replaced by a
description produced by a vision model, so the text-only model gets something
to work from:

```
[image — /home/you/.pi/agent/cache/image-compact/eb49a844ca83324e.png]
The active model cannot accept images, so a vision model described this one instead.
If the description is not specific enough, call inspect_image with the path above and a focus.
Do not assume it is complete.

Error: ENOENT

no such file or directory

open config.yaml
```

## Why not `before_provider_request`

It is the obvious hook and it cannot work. `downgradeUnsupportedImages` runs
first, so by the time that event fires the base64 is gone and only the
placeholder string remains. There is nothing left to describe. Any
implementation built on it is a silent no-op.

The conversion therefore happens where the bytes still exist:

| Hook | Covers | When it runs |
|---|---|---|
| `tool_result` | the read tool returning a png/jpg/gif/webp/bmp | once; the image never enters history |
| `context` | an image pasted or attached by the user | before every LLM call, cached after the first |

## The trigger, and a trap

`ctx.model.input` at the moment of the call. No catalog is parsed, so providers
registered at runtime — the `opencode-accountN` ones from multi-account — are
handled.

The tempting move is to keep an exception list for models whose catalog entry
lies. Don't. `opencode/muse-spark-1.3-contributor-free` is catalogued
`input: ["text"]` yet reads images correctly — three runs in a row, no extension
loaded. It also reports no image support on other runs of the identical
command, and pi strips the image. A free tier that load-balances across
backends will not agree with itself about its own modalities.

So the flag is unreliable as a *description of the model* but reliable as a
*description of the request*: when it says the model cannot take images, pi is
about to drop the image, so converting is strictly better than losing it.
Exempting a model does not upgrade it — it guarantees the image vanishes on
every run where the flag is pessimistic.

## Cost

One nested `pi` call per image, on a vision model, reusing the credentials pi
already holds. No API key is read, stored or forwarded by this file. Results
are cached by content hash under `~/.pi/agent/cache/image-compact/`, so a
screenshot costs one call ever, not one per turn.

The child needs the same free-tier treatment as any other request: the
`opencode-free-tier` patch, `multi-account` for the account providers, and the
full builtin tool list. A read-only child is rejected with a 403, because the
free tier checks the tool list in the request.

## `inspect_image`

Resending the image would not help a model that cannot decode it, so the
fallback sends a vision model back to the same file with a focus:

```
inspect_image({ path: "...png", focus: "the error dialog in the top right" })
```

That is strictly more useful than handing back pixels the model cannot read.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PI_IMAGE_COMPACT_MODEL` | `opencode-account1/muse-spark-1.3-contributor-free` | which model describes the image |
| `PI_IMAGE_COMPACT_FORCE` | unset | `1` converts even for models that accept images |
| `PI_IMAGE_COMPACT_TIMEOUT_MS` | `120000` | per-image timeout |
| `PI_IMAGE_COMPACT_CHILD_EXTENSIONS` | multi-account, opencode-free-tier | comma-separated paths loaded into the child |
| `PI_IMAGE_COMPACT_CHILD_TOOLS` | `read,write,edit,bash,grep,find,ls` | child tool list |

## Install

```
pi install git:github.com/hakergeniusz/pi-image-compact
```

MIT.
