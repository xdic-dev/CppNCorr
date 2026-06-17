# CppNCorr User Guide

This guide covers running the CppNCorr DIC tool (`proxyncorr`): supported image
formats and naming conventions, every CLI flag, the config-file format, the
output layout, and troubleshooting.

## Supported image formats and naming conventions

### Formats

`proxyncorr` reads the image formats OpenCV's `imread` supports in this build:

| Extension | Format |
|-----------|--------|
| `.png` | PNG |
| `.tif`, `.tiff` | TIFF |
| `.bmp` | BMP |
| `.jpg`, `.jpeg` | JPEG |

Extension matching is **case-insensitive**.

### Folder layout and naming

`proxyncorr` discovers frames from a folder. The expected layout:

- **One image per frame.** Frames are sorted in *natural* (numeric-aware) order,
  so both zero-padded (`frame_00.png`, `frame_01.png`, ...) and unpadded
  (`frame_2.png`, `frame_10.png`, ...) numbering sort correctly.
- **ROI mask.** A `roi.png` in the folder is used as the region-of-interest mask
  by default (any pixel > 0.5 grayscale is inside the ROI). Override with
  `--roi`.
- **Reference frame.** If a `ref.png` exists in the folder it is used as the
  reference; otherwise the first discovered frame is the reference. Override with
  `--ref`.
- **Reserved / ignored.** `roi.png` and `ref.png` (case-insensitive) are excluded
  from the frame list, as are any files matching the `--roi` / `--ref`
  basenames, hidden files (leading `.`), sub-directories, and non-image
  extensions.

## CLI flags

Every flag below is accepted by `proxyncorr`. Types and defaults match the
compiled defaults (which themselves come from `ncorr::Config` and the config
file; CLI args have the highest priority).

### Paths

| Flag | Short | Type | Default | Example |
|------|-------|------|---------|---------|
| `--folder` | `-f` | path | `images` | `--folder data/run1` |
| `--config` | `-c` | path | (none; falls back to `config/default.cfg` if present) | `--config my.cfg` |
| `--roi` | `-r` | path | `<folder>/roi.png` | `--roi masks/roi.png` |
| `--ref` | `-R` | path | first frame (or `<folder>/ref.png`) | `--ref data/ref.png` |
| `--output` | `-o` | path | `output` | `--output out/` |

### DIC parameters

| Flag | Short | Type | Default | Example |
|------|-------|------|---------|---------|
| `--scalefactor` | `-s` | int | `3` | `--scalefactor 1` |
| `--interp` | `-i` | enum | `QUINTIC_BSPLINE_PRECOMPUTE` | `--interp CUBIC_KEYS` |
| `--subregion` | `-S` | enum | `CIRCLE` | `--subregion SQUARE` |
| `--radius` | `-d` | int | `20` | `--radius 30` |
| `--threads` | `-t` | int | `4` | `--threads 8` |
| `--mode` | `-m` | enum | `auto` | `--mode parallel` |

`--interp` accepts: `NEAREST`, `LINEAR`, `CUBIC_KEYS`, `CUBIC_KEYS_PRECOMPUTE`,
`QUINTIC_BSPLINE`, `QUINTIC_BSPLINE_PRECOMPUTE`.
`--subregion` accepts: `CIRCLE`, `SQUARE`.
`--mode` accepts: `auto`, `sequential`, `parallel` (`auto` picks `sequential`
when seeds are supplied, otherwise `parallel`).

### Seeds

| Flag | Type | Default | Example |
|------|------|---------|---------|
| `--seeds` | path | (none) | `--seeds seeds.json` |
| `--seeds-optimized` | switch | off | `--seeds-optimized` |

`--seeds` points to a JSON array of seed objects, one per region:

```json
[
  {"x": 100, "y": 200, "u": 0.0, "v": 0.0},
  {"x": 300, "y": 400, "u": 0.5, "v": 0.3}
]
```

`--seeds-optimized` declares the supplied seeds are already optimized (skips the
optimization step).

### Units and strain

| Flag | Short | Type | Default | Example |
|------|-------|------|---------|---------|
| `--units` | `-u` | string | `mm` | `--units px` |
| `--units-per-pixel` | `-p` | double | `0.2` | `--units-per-pixel 0.05` |
| `--strain-subregion` | | enum | `CIRCLE` | `--strain-subregion SQUARE` |
| `--strain-radius` | | int | `5` | `--strain-radius 7` |

### Video / output toggles

| Flag | Short | Type | Default | Example |
|------|-------|------|---------|---------|
| `--alpha` | `-a` | double | `0.5` | `--alpha 0.3` |
| `--fps` | `-F` | double | `15` | `--fps 30` |
| `--no-json` | | switch | (JSON on) | `--no-json` |
| `--no-binary` | | switch | (binary on) | `--no-binary` |
| `--no-videos` | | switch | (videos on) | `--no-videos` |
| `--debug` | | switch | off | `--debug` |

### Post-DIC output flags

| Flag | Type | Default | Behaviour |
|------|------|---------|-----------|
| `--export-video` | switch | off | Forces video output on (renders displacement/strain fields). |
| `--export-strains` | switch | off | Ensures strain fields are written (enables binary output if both JSON and binary were disabled). |
| `--change-perspective` | enum | `eulerian` | `eulerian` is implemented (the default). `lagrangian` and other modes print "not yet implemented" and exit cleanly. |
| `--help` | switch | — | Print usage and exit. |

### Example

```bash
./test/bin/proxyncorr \
    --folder test/examples/ohtcfrp/images \
    --output test/examples/ohtcfrp \
    --scalefactor 1 --radius 30 --threads 4 \
    --units mm --units-per-pixel 0.2 \
    --no-videos
```

## Config file format and placement

The config file is INI (`key = value`). It is loaded when passed with
`--config <path>`; if no `--config` is given, `proxyncorr` falls back to
`config/default.cfg` (relative to the working directory) when that file exists.

```ini
# Comments start with # or ;   (inline comments are supported)
scalefactor = 3
subregion_type = CIRCLE
subregion_radius = 20
interp_type = QUINTIC_BSPLINE_PRECOMPUTE
strain_subregion_type = CIRCLE
strain_radius = 5
dic_config = NO_UPDATE
num_threads = 4
debug = false
perspective_interp = CUBIC_KEYS
units = mm
units_per_pixel = 0.2
alpha = 0.5
fps = 15.0
```

Override precedence (highest wins): **CLI args > config file > compiled
defaults**. Key names must match `ncorr::Config` field names exactly. See the
full schema in the [developer guide](developer_guide.md#config-file-schema-reference).

## Output directory structure

For `--output out/`, the tool creates:

```
out/
├── save/                 # binary serialized data (when binary output enabled)
│   ├── DIC_input.bin
│   ├── DIC_output.bin
│   ├── strain_input.bin
│   └── strain_output.bin
├── save_json/            # same data as JSON (when JSON output enabled)
│   ├── DIC_input.json
│   ├── DIC_output.json
│   ├── strain_input.json
│   └── strain_output.json
└── video/                # rendered field videos (when video output enabled)
    ├── u_eulerian.avi
    ├── v_eulerian.avi
    ├── exx_eulerian.avi
    ├── eyy_eulerian.avi
    └── exy_eulerian.avi
```

The JSON `DIC_output.json` contains per-frame `disps` with `u`/`v` arrays (each
with `rows`/`cols`/`data`), the ROI mask, the scale factor, units, and
units-per-pixel. The strain JSON contains `exx`/`eyy`/`exy` fields per frame.

## Troubleshooting

1. **`Error: ROI file not found: <folder>/roi.png`**
   The tool needs a ROI mask. Put a `roi.png` in the image folder or pass
   `--roi path/to/roi.png`.

2. **`Error: No frames found in folder: <folder>`**
   The folder has no supported image files (after excluding `roi.png`/`ref.png`
   and hidden files/sub-directories). Check the path and that frames use a
   supported extension (`.png/.tif/.tiff/.bmp/.jpg/.jpeg`).

3. **`Error: Cannot open folder '<folder>': ...`**
   The `--folder` path does not exist or is not readable. Verify the path
   (paths are resolved relative to the current working directory).

4. **Frames analysed in the wrong order.**
   Ordering is natural/numeric-aware, but only across the numeric runs in the
   names. Use a consistent numbering scheme (e.g. `frame_00`, `frame_01`, ...).
   Mixed prefixes can sort unexpectedly; rename to a uniform pattern.

5. **`Error in config file '...': INI value for '<key>' is not a valid integer/number`**
   A config value has the wrong type (e.g. text where a number is expected), or
   a key is malformed. Fix the value; key names must match `ncorr::Config`
   fields exactly.

6. **`[--change-perspective=<mode>] not yet implemented` and the tool exits.**
   Only `eulerian` is currently implemented. Use the default (omit the flag) or
   pass `--change-perspective eulerian`.

7. **Run is very slow / uses one core.**
   Increase `--threads`, lower the `--scalefactor`, or reduce the number of
   frames. Real DIC is compute-heavy; the bundled `ohtcfrp` fixture is small but
   still takes tens of seconds per frame at fine settings. Add `--no-videos` to
   skip video rendering.
