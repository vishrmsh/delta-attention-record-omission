# Packaging verification

This release was checked on 11 September 2026.

- Every shipped file matches the inventory, size, and SHA-256 checksum.
- Every Python source parses and every JSON file passes strict parsing.
- The 40 forcing, 40 sampling, and 36 family reports regenerate the shipped
  summary JSONs with complete equality; the numerical TeX macros regenerate
  byte for byte, without pending values.
- The Qwen manifest and measurement implementation retain the recorded hashes.
- All shipped measurement implementations match the maintained source files.
  Two package initializers remain intentionally minimal to avoid unrelated
  application imports.
- The quantitative figure renderer is the current maintained version. It
  changes labels, composition and presentation, with all measurement records
  and numerical macros unchanged. Previous geometry-preservation statements
  applied to an earlier renderer revision and do not describe this update.
- The five existing Qwen publication-validation tests passed.
- An inventory and text scan found no creator identity, email address, private
  source-repository URL, absolute creator path, symlink, credential file,
  checkpoint, or new text-bearing clinical export. All evidence files are
  byte-identical to the previously reviewed source-free artifact.

The only HTTP URL in the implementation code is the XML namespace used by
the figure sanitizer. Source-text exceptions remain limited to the invented
and public-text examples documented in the README.

No pretrained models were loaded, no controlled datasets were read, and no
model measurements were repeated. Verification concerns artifact integrity and
aggregation, not independent replication of the measurements. Experiments
still in progress are excluded from this release.
