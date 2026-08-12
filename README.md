# Segmentation mask correction tool

This package serves paired TIFF images and instance-segmentation masks to a
browser-based Kaibu editor through Hypha. The image data remain on the machine
running the Python process and are sent to the collaborator's browser on demand.

## Expected files

For every image, the directory must contain a mask with the same basename:

```text
sample.tif
sample_masks.tif
```

Masks must be 2-D integer label images: `0` is background and every positive
integer identifies one instance. Subdirectories are supported.

The editor writes corrections beside the source files as:

```text
sample_masks_corrected.tif
```

Original `*_masks.tif` files are never overwritten. If a corrected file already
exists, it is loaded and updated on the next editing session.

## Run it

From this repository:

```bash
uv run serve /links/shared/scuanalysis/Hierlemann/Marta/data/run1/training
```

uv creates and updates the project environment from `pyproject.toml` and
`uv.lock`; no separate installation step is needed.

The command prints a long ImJoy URL. Send that complete URL to the collaborator
and keep the terminal process running for the duration of the session. No inbound
port or tunnel is needed: both the Python process and browser connect outbound to
the Hypha relay. Each launch adds a cache-busting version to the plugin URL so a
newly deployed interface is not replaced by a stale browser or CDN copy.

To choose another output suffix:

```bash
uv run serve DIRECTORY --corrected-suffix _masks_reviewed.tif
```

Share links expire after 24 hours by default. To choose another lifetime, while
the serving process remains running:

```bash
uv run serve DIRECTORY --link-expiry-hours 48
```

## Editor workflow

- Each colored, translucent polygon in the **Instances** layer represents one
  instance label. Colors are deterministic and visually separate neighboring
  labels; fills use 25% opacity with a thin outline.
- Use the selection tool to move contour vertices. Draw a closed path in the
  Instances layer to add a missing instance.
- Select a contour and use **Delete**, **Backspace**, or Alt/Shift+D to remove it.
- Keyboard shortcuts are **D** for draw mode, **S** to save, **N** to open the
  next image, and **R** to discard unsaved changes and reload. Kaibu's
  Ctrl/Cmd+Z undo and arrow-key movement shortcuts remain available.
- The **Samples** tab shows the sample hierarchy as a tree. Double-clicking a
  sample changes images without saving the current correction.
  Compact editing controls are available in the **Actions** tab, and support
  details and a link to the GitHub repository are listed under **Info**.
- **Next** discards unsaved changes and opens the next image.
- **Reload** discards unsaved browser changes and reloads the last saved mask.
- Corrected masks are written only when **Save** or the **S** shortcut is used.
- Scalar images are explicitly rendered with the grayscale color map.

Stop the Python process with Ctrl-C when the collaborator is finished. The share
URL contains a temporary workspace token, so share it only with the intended
collaborator.

## Filename options

The defaults can be changed with `--mask-suffix` and `--corrected-suffix`. Run
`uv run serve --help` for all options, including selection of a different Hypha
server or ImJoy plugin URL.
