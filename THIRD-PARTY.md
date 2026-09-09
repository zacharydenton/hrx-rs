# Native distribution status

The MIT license in this repository covers the Rust code. It does not grant a
license to redistribute the native bundle's dependencies. The current bundle is
not cleared for publication: its inherited dependency builds have unverified
provenance, and it contains no complete dependency notices or license texts.
`Cargo.toml` disables publication while this remains unresolved.

`THIRD-PARTY.json` and `NOTICE` identify every shipped binary from evidence in
the binaries themselves: GNU build-ids, `DT_SONAME`, `DT_NEEDED`, `.comment`
toolchain records, debuglink names and embedded version strings. Both files are
marked incomplete, because identification is not licensing. No component has a
confirmed SPDX identifier and no upstream license texts have been collected.

Identification found three distinct origins, which the inventory keeps separate:

| Origin | Shipped components | Remaining work |
| --- | --- | --- |
| Built here from `ROCm/hrx-system` + `patches/loom` | libhrx.so, libhrx.so.0, libloomc.so | Upstream license, plus every statically linked dependency |
| AMD ROCm release build (`rockrel` CI paths remain in the binaries) | libhsa-runtime64.so.1 (ROCR-Runtime 1.21.0), librocm_sysdeps_{elf,numa,drm,drm_amdgpu,z,zstd,liblzma,bz2} | Licenses and notices from upstream sources; elfutils, libnuma and libdrm versions from the AMD build record |
| Arch Linux packages swept in from the build host | librocprofiler-register.so.0 (0.6.0), libfmt.so.12 (12.2.0), libglog.so.2 (0.7.1), libgflags.so.2.2 (2.2.2) | Replace with the matching AMD build; see below |

The third row is the significant finding. Those four binaries are byte-identical
by GNU build-id to the copies installed on the machine that assembled the
bundle. They are not artifacts of the AMD build that produced
`libhsa-runtime64.so.1`. Only `librocprofiler-register.so.0` is actually
required, as a `DT_NEEDED` of the HSA runtime; `libfmt` and `libglog` exist only
to satisfy it, and `libgflags` only to satisfy `libglog`. Sourcing
rocprofiler-register from the same AMD build as the HSA runtime may remove all
four from the bundle, and with them four entries from this inventory.

Before publishing a replacement bundle:

1. Rebuild inherited binaries from identified sources, or recover verifiable
   build records. Record exact source revisions, patches, build commands and
   binary digests in `provenance.json`, including statically linked dependencies.
2. Complete the `THIRD-PARTY.json` inventory: it already maps each binary to its
   source, version and evidence, but every `license.status` is `unconfirmed`.
   Fill in license identifier(s) and included license text filenames from the
   upstream source distributions, not from the binaries. Preserve required
   copyright and attribution notices in `NOTICE`; include all applicable license
   texts and required source materials or source-distribution arrangements.
   Review these against the actual sources.
3. Stage those files alongside the libraries. Bundles are flat: use names such
   as `LICENSE-libfmt.txt`. `hrx pack` requires nonempty provenance, inventory and
   notice files and hashes all staged regular files into its manifest. Its
   structural checks do not perform a legal or provenance review.
4. Publish the reviewed archive at an anonymously accessible HTTPS URL, update
   `bundle.json`, and verify installation into an empty cache plus the complete
   ignored test suite. Only then remove `publish = false`.

