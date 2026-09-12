# Conformance corpus

Static catalogs of specification cases, with the section of the
standard each one comes from. They are checked in rather than fetched,
so a test run never depends on the network or on an upstream repository
staying available.

| File | Standard | Cases |
|---|---|---|
| `rfc8259.json` | RFC 8259 (JSON, STD 90) | 347 |
| `rfc7493-ijson.json` | RFC 7493 (I-JSON) | 6 |

Run them with:

```bash
pixi run mojo -I . tests/test_conformance.mojo
```

`tests-cpu` runs the same file, so a regression fails CI.

## Catalog shape

Each file is one standard:

```json
{
  "format": "json",
  "standard": "RFC 8259",
  "version": "8259",
  "standard_url": "https://www.rfc-editor.org/rfc/rfc8259",
  "cases": [
    {
      "id": "json-8259-empty-object",
      "title": "Empty object is a JSON text",
      "section": "2",
      "section_url": "https://www.rfc-editor.org/rfc/rfc8259#section-2",
      "requirement": "MUST",
      "expect": "accept",
      "input": "{}",
      "input_encoding": "utf-8",
      "decoded": {}
    }
  ]
}
```

- `expect` is `accept`, `reject`, or `any`. An `any` case is one the
  standard leaves to the implementation; the runner reports what we do
  with it and never fails on it.
- `input_encoding` is `utf-8` (the default) or `hex`, the latter for
  inputs that are not valid UTF-8 or not text at all. Hex inputs reach
  the parser as raw bytes.
- `decoded`, when present, is the value the input must parse to.

## Provenance

`rfc8259.json` and `rfc7493-ijson.json` were assembled by the
GLD.SerializerBenchmark project (MIT), which merged the JSONTestSuite
parsing corpus with original section-linked cases. `jts-` ids are the
JSONTestSuite cases; `json-` and `ijson-` ids are original. See
`LICENSE-JSONTestSuite` for the upstream notice.

## Reading a failure

The runner prints one line per failing case with the section URL, so
the first thing to do is open that URL and read the rule. A case may be
wrong -- if so, say why in the test file rather than deleting the case.
