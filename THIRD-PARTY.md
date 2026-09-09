# Native distribution status

The MIT license in this repository covers the Rust code. It does not grant a
license to redistribute the native bundle's dependencies. The current bundle is
not cleared for publication: its inherited dependency builds have unverified
provenance, and it contains no complete dependency notices or license texts.
`Cargo.toml` disables publication while this remains unresolved.

The unresolved inventory includes:

| Shipped component | Required provenance and notice review |
| --- | --- |
| libhsa-runtime64.so.1 | HSA runtime source revision, patches and build recipe |
| libglog.so.2, libgflags.so.2.2, libfmt.so.12 | Exact dependency sources and notices |
| librocm_sysdeps_zstd.so.1, librocm_sysdeps_liblzma.so.5 | Exact compression library sources and notices |
| librocm_sysdeps_drm.so.2, librocm_sysdeps_drm_amdgpu.so.1 | Exact libdrm sources and notices |
| librocm_sysdeps_elf.so.1, librocm_sysdeps_numa.so.1 | Exact library sources and redistribution requirements |
| librocm_sysdeps_bz2.so, librocm_sysdeps_z.so.1 | Exact compression library sources and notices |
| librocprofiler-register.so.0 | Source revision, build recipe and notices |
| libhrx.so, libloomc.so, loom-compile | Recorded upstream revision and patches, plus all linked dependencies |

Before publishing a replacement bundle:

1. Rebuild inherited binaries from identified sources, or recover verifiable
   build records. Record exact source revisions, patches, build commands and
   binary digests in `provenance.json`, including statically linked dependencies.
2. Supply a `THIRD-PARTY.json` inventory mapping each binary and linked component
   to its source, version/revision, license identifier(s), and included license
   text filenames. Preserve required copyright and attribution notices in
   `NOTICE`; include all applicable license texts and required source materials
   or source-distribution arrangements. Review these against the actual sources.
3. Stage those files alongside the libraries. Bundles are flat: use names such
   as `LICENSE-libfmt.txt`. `hrx pack` requires nonempty provenance, inventory and
   notice files and hashes all staged regular files into its manifest. Its
   structural checks do not perform a legal or provenance review.
4. Publish the reviewed archive at an anonymously accessible HTTPS URL, update
   `bundle.json`, and verify installation into an empty cache plus the complete
   ignored test suite. Only then remove `publish = false`.

