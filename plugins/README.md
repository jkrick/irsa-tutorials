# IRSA MyST Plugins

Custom [MyST](https://mystmd.org) plugins for the irsa-tutorials.

## Building

```bash
cd plugins
npm install
npm run build
```

This bundles `src/notebook-gallery.mjs` into `dist/notebook-gallery.mjs`. Must *re-run* after any changes to `src/`.

## Registering with MyST

Add the built bundle to `myst.yml`:

```yaml
project:
  plugins:
    - plugins/dist/notebook-gallery.mjs
```

---

## Plugin: `notebook-gallery` directive

Renders a list of notebooks as a responsive grid of clickable cards. Each card shows a title and a one-sentence description pulled from a shared metadata file.

### Metadata file

The directive argument is a path to a YAML or JSON file (relative to the repo root) that lists notebook metadata. Each entry must have at least:

| Field         | Type   | Description                                          |
| ------------- | ------ | ---------------------------------------------------- |
| `file`        | string | Path to the notebook relative to the repo root       |
| `title`       | string | Display title shown in the card header               |
| `description` | string | One-sentence summary shown in the card body          |

Example `notebook_metadata.yml`:

```yaml
- file: tutorials/wise/wise_catalog_search.md
  title: WISE Catalog Search
  description: Search the AllWISE catalog using astroquery.
- file: tutorials/euclid/1_Euclid_intro_MER_images.md
  title: Euclid MER Images
  description: Access and visualise Euclid Q1 MER mosaic images.
```

### Directive syntax

````markdown
```{notebook-gallery} notebook_metadata.yml
tutorials/wise/wise_catalog_search.md
tutorials/euclid/1_Euclid_intro_MER_images.md
```
````

- The **argument** (on the opening fence line) is the path to the metadata file.
- The **body** lists one notebook path per line (matching the `file` field in the metadata). Lines starting with `#` are treated as comments.

### Error handling

| Situation                           | Rendered output                                          |
| ----------------------------------- | -------------------------------------------------------- |
| Metadata file not found / unreadable | An `{error}` admonition with the bad path               |
| Notebook path not in metadata        | A warning card: _⚠️ Unrecognised notebook_              |
