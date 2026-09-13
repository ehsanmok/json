# Conformance corpus

Static catalogs of specification cases, with the section of the
standard each one comes from. They are checked in rather than fetched,
so a test run never depends on the network or on an upstream repository
staying available.

| File | Standard | Cases | Runner |
|---|---|---|---|
| `rfc8259.json` | RFC 8259 (JSON, STD 90) | 347 | `test_conformance.mojo` |
| `rfc7493-ijson.json` | RFC 7493 (I-JSON) | 6 | `test_conformance.mojo` |
| `rfc6901-pointer.json` | RFC 6901 (JSON Pointer) | 18 | `test_pointer_patch_conformance.mojo` |
| `rfc6902-patch.json` | RFC 6902 (JSON Patch) | 26 | `test_pointer_patch_conformance.mojo` |
| `rfc7396-merge-patch.json` | RFC 7396 (JSON Merge Patch) | 21 | `test_pointer_patch_conformance.mojo` |
| `jsonschema-2020-12.json` | JSON Schema draft 2020-12 | 197 | `test_schema_conformance.mojo` |

Run the grammar catalogs with:

```bash
pixi run mojo -I . tests/test_conformance.mojo
```

`tests-cpu` runs that file, so a regression fails CI.

Run the pointer and patch catalogs with:

```bash
pixi run -e dev mojo run -I . tests/test_pointer_patch_conformance.mojo
```

That one is deliberately outside `tests-cpu`. It reports rather than
asserts: every failing case is sorted into one of four root causes
(`accepted-invalid`, `wrong-result`, `not-atomic`, `rejected-valid`) and
printed with the section URL and what the library actually did. Wiring
it into CI is the last step of closing those gaps, not the first.

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

## Pointer and patch shape

The three layered-specification files keep the same envelope -- `format`,
`standard`, `version`, `standard_url`, `cases`, and a `section_url` per
case -- and differ in what a case holds, because a pointer case has one
input and a patch case has two. Each file names its own case fields in a
`field_guide` object at the top, so the file is readable without this
README. In summary:

| File | Case fields |
|---|---|
| `rfc6901-pointer.json` | `document`, `pointer`, `expect` (`resolve`/`error`), `result` |
| `rfc6902-patch.json` | `document`, `patch` or `patch_text`, `expect` (`apply`/`error`), `result`, `check_unchanged` |
| `rfc7396-merge-patch.json` | `kind` (`merge`/`diff`), then `original`/`patch`/`result`, or `source`/`target`/`expected_patch` |

Two of those need a word beyond the field guide:

- `patch_text` holds a patch that cannot be written as parsed JSON. The
  only case that needs it is RFC 6902 A.13, whose operation object names
  `op` twice; a corpus storing it as parsed JSON would have had to pick
  one of the two and lose the case.
- A `diff` case asserts only the round trip, that applying the generated
  patch to `source` yields `target`. RFC 7396 section 3 does not make any
  one generator output normative, so `expected_patch` is reported as
  informational when it differs.

Results are compared member-order-insensitively, and `1` compares equal
to `1.0`, because RFC 8259 gives object members no order and RFC 6902
section 4.6 defines number equality by value.

## Provenance

`rfc8259.json` and `rfc7493-ijson.json` were assembled by the
GLD.SerializerBenchmark project (MIT), which merged the JSONTestSuite
parsing corpus with original section-linked cases. `jts-` ids are the
JSONTestSuite cases; `json-` and `ijson-` ids are original. See
`LICENSE-JSONTestSuite` for the upstream notice.

`rfc6901-pointer.json`, `rfc6902-patch.json` and
`rfc7396-merge-patch.json` were transcribed from the RFC texts
themselves, not taken from any third-party suite. Each of those three
specifications carries its own worked examples, and those examples are
the bulk of each file:

- RFC 6901 section 5 gives one document and the twelve pointers that
  address every member of it.
- RFC 6902 appendix A is sixteen numbered example patches, A.1 through
  A.16, transcribed here in order.
- RFC 7396 appendix A is a fifteen-row table of original document, patch
  and result, transcribed here in order as `merge-a01` through
  `merge-a15`.

The remaining cases in each file are original, and come from the
normative text rather than the examples: the syntax and evaluation rules
in RFC 6901 sections 3 and 4, the per-operation rules in RFC 6902
section 4 and the atomicity rule in its section 5, and the pseudocode in
RFC 7396 section 2. They are the rules an implementation can pass every
published example while still getting wrong, which is why they are here.
Every case carries the `section_url` it was read from, so a disputed
case can be checked against the source in one click.

## Reading a failure

The runner prints one line per failing case with the section URL, so
the first thing to do is open that URL and read the rule. A case may be
wrong -- if so, say why in the test file rather than deleting the case.
