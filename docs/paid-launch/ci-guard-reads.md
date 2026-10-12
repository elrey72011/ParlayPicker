# Live scope-read optimization

Pinned sequential nine-case profiles preserve the three original scope files,
their assertions, isolated copies, network blocking and rejection paths.
Main before: 493.145s/17,081 Git launches. Initial PR: 501.030s/17,081.
Object stream plus exact immutable marker caching: 200.654s/3,852.
At the isolated-bootstrap seal, explicitly enabled transport alone: 202.661s;
fresh traversal prototype: 185.919s/3,852; policy-construction prototype:
111.755s/1,042. These are completed local measurements, not CI forecasts.
Raw profiles, failed/superseded CI and the bootstrap failure remain retained.

The measured remainder includes standalone Git policy-construction reads and
repeated pathlib directory traversal. Exact-source policy constructors are
wrapped in one fresh readonly object stream per invocation. Their original
files and functions remain unchanged. The fixed registry is under 64 modules;
only already-loaded modules at the consumed guard location with exact Git blob
identities are eligible. Source bytes (bounded at 128 KiB) are reverified at
each constructor call. Outputs, HEAD, index, filesystem facts and decisions
are never cached. Existing helper/source bytes stay available unchanged.

LiveRoot performs a fresh scandir traversal for each basename query. Unsupported
patterns/options use pathlib. Hidden files and directories remain included;
directory symlinks are not followed. Every validation invocation restores ROOT
in finally. Exact-source predecessor modules receive readonly transport only;
their reconstruction, hashes and actual validation logic stay unchanged.

All validation entrypoints use lazy hash-verified helper loading: an isolated
guard can reconstruct historical bytes with no adjacent helper. Actual
validation cannot skip a missing/changed helper, binding or policy. The new
two-entry pure reconstruction cache is capped at 8 MiB per source, at most
32 MiB conservative input/output; together with the prior new caches, at most
160 MiB. Existing frozen caches are unchanged. No mutable decision is cached.

Acquisition, models, original tests, baselines, scheduler/profile, workflows,
timeouts and wagering protections are unchanged. Fresh final-head CI with
complete reports and exact executed partition is required; cancelled progress
is not a completed measurement or a passing suite.
