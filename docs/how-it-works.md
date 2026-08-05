# How it works

SAM 2's video predictor is built for short, pre-recorded clips: it decodes the whole video into one
big tensor before tracking. Each frame, at the model's input size, is about 12 MB, so a few thousand
frames is already tens of gigabytes — too much for a live stream (which never ends) or a long/4K clip.

Two things grow with video length. `sam2-stream` bounds both.

## 1. Frames: a ring buffer

While tracking, SAM 2 only ever reads the **current** frame. It never goes back to re-read old raw
pixels — its memory of the past is a compact set of encoded features, not the frames themselves.

So we keep a fixed number of frame slots (default 32) and overwrite the oldest as new frames arrive.
The frame index keeps counting up (SAM 2's temporal attention needs the true distance between
frames), but the storage slot wraps around:

```
slot = frame_index % width
```

With a 4-slot buffer as an example:

| frame arriving | slot (`% 4`) | note |
|----------------|--------------|------|
| 0 | 0 | first four frames fill the slots |
| 1 | 1 | |
| 2 | 2 | |
| 3 | 3 | |
| 4 | 0 | overwrites frame 0 |
| 5 | 1 | overwrites frame 1 |
| … | … | only ever 4 frames stored |

The predictor is told the video is very long (a large `virtual_len`), so it keeps advancing, but only
`width` frames ever exist in memory. Each fresh frame is written into its slot just before the
predictor reaches it.

## 2. Tracker memory: eviction

For every frame it processes, SAM 2 stores a small summary so it can recognize the object next frame.
Left alone, that store grows by one entry per frame forever.

But SAM 2 only attends to roughly the last 7 frames of that memory (plus the frame you prompted on).
Older entries are never read, so we delete anything older than a short window (default 16) each step.
The prompt frame is a conditioning frame and is kept, so tracking is unaffected.

## Result

| | naive (load everything) | sam2-stream |
|---|---|---|
| frames in memory | all of them (grows) | fixed ring (constant) |
| tracker memory | grows every frame | short window (constant) |
| total | grows until it runs out | flat, regardless of length |

Everything that used to grow with video length is now a fixed size, so memory stays flat whether you
run for ten seconds or ten hours.

## A note on memory horizon

Because the memory window is measured in **frames**, the amount of real time it spans depends on your
frame rate. At 30 FPS, ~7 frames is a fraction of a second; at 5 FPS it is over a second. If you need
to bridge longer gaps (an object leaving and returning), feeding fewer frames per second stretches the
window in wall-clock time, at the cost of finer motion detail. SAM 2 is still a continuous tracker,
not a re-identification model — an object gone for a long time generally needs to be re-seeded.
