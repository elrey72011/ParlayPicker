# Owner-only NFL running-build preflight

Authorized successor of cleared main3fb9302463f44589ea0d2c75b2ed100bf2f596d0,
treecdc34e454f7b5312e2673ecfd39126956efe5d39. One conditional PR cycle follows
complete local qualification. Owner merges; no deployment or database invocation
before actual main and running-build verification. The existing diagnostic
allowance remains unused; build reads neither consume nor reset it.

Existing Cloud records identify the repository, main branch, fresh clone/pull
and process startup, but omit the resolved commit. GitHub deployment records
are not required. The preflight therefore observes local Git commit/tree,
compares relevant physical source with Git bytes, then compares compiled source
with actual loaded code objects, literal constants and imported Store/service
aliases. Compilation never executes source. Missing or conflicting evidence
fails closed. This is scoped code evidence, not attestation of all interpreter
memory, generated dataclass methods, packages or the operating system.
Source/tree reads are anchored to the observed commit. HEAD is checked again
at the end; a checkout move makes the observation fail closed.
Git object reads disable lazy fetching and network protocols. Missing local
objects fail instead of causing an implicit network acquisition.
Supported function defaults are independently checked against syntax and live
global aliases: code objects alone cannot detect a changed default argument.
Unsupported defaults fail closed. Literal comparisons reject custom objects
instead of invoking their equality methods. Retain the focused stale-default
false-acceptance reproduction and its regression.

The explicit preflight button is separate from the read-only connection button,
inside the existing owner-token gate. It reads no DSN, hostname, key or private
setting. No socket, Store construction, verification, provider or predictor is
called. It reports observed runtime versions and process identity. Linux process
start comes from procfs; unavailable start evidence remains UNKNOWN.

Binding plan: preserve prior qualification and diagnostic policies unchanged.
Extend the current qualification dispatcher with an authenticated narrow
successor. Store its metadata inside the existing qualification record, so the
frozen capacity parent reconstruction remains byte-identical. Preserve original
predecessor expectations; new beforeimages authenticate exact prior sources.
The original capacity fixture also consumes the diagnostic UI at generation
f6a7506. Its beforeimage is retained from exact cleared main (unchanged since
that generation), independently pinned in a regression. Without this producer
edge its original authentication correctly rejects the new UI. No historical
fixture, frozen parent component or expected rejection is changed.
All existing application/model/store/transport/replay/watchdog bytes and limits
remain unchanged. Validate actual PR/push scope, shallow historical consumers,
owner UI and zero-connection regressions before the sole publication cycle.

The complete owner UI context demonstrated NFL_SCENARIO_STARTED after the fixed
synthetic kickoff passed in real time: preparation/inference used the fixture
clock but native completion used the real clock. The successful synthetic-flow
helper now binds only that completion clock to the same provider clock, restored
by monkeypatch teardown. All original assertions, fixed midnight fixtures,
real pregame rejection, model and production caller remain unchanged. Preserve
the original failures and qualify the complete context after the correction.

After owner merge: reconcile automatic main outcomes and reviewed-tree equality;
deploy that cleared revision and at most one controlled restart if required.
Read build information without enabling the normal workflow. Compare observed
commit/tree and relevant loaded-code evidence with the cleared revision and
deployment/process-start records. Only then may the existing separately tracked
one-use storage diagnostic run. An interrupted or failed storage invocation is
spent, regardless of a new session or process. Keep both NFL switches false.
