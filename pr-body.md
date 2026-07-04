## Description of what I changed

MariaDB 11 dropped the `5.5.5-` version-string prefix, so Liquibase now detects it as MariaDB rather than MySQL and silently skips every changeset declared with `dbms="mysql"`. On MariaDB 11 deployments this leaves real schema and core data missing: the `cohort_member` primary-key/uuid migrations never run (and the unique constraint on `cohort_member.uuid` then fails on populated tables), `scheduler_task_config` uuids are never generated before their not-null constraint, the `System Developer` role is never re-created, and 248 further changesets in the 2.0.x update log plus the `FOREIGN_KEY_CHECKS` toggles in the 1.9.x core-data snapshot never run.

**The design question that was open on the ticket — whether editing historical changesets breaks checksum validation on existing databases — was answered empirically** against Liquibase 4.32.0 (the version resolved by this repo), for both v8 and v9 checksum versions:

| Edit | Checksum effect |
|---|---|
| `dbms="mysql"` → `dbms="mysql,mariadb"` on the changeSet element | **unchanged** |
| adding `<dbms type="mariadb"/>` inside `<preConditions>` | **unchanged** |
| adding a `<modifySql dbms="mariadb">` block | **changed** |

(A control mutation of real changeset content was verified to change the checksum, confirming the measurement setup detects drift.)

The fix follows directly from those measurements:

1. **In-place `dbms="mysql,mariadb"` extension** for 248 changesets in `liquibase-update-to-latest-2.0.x.xml`, 3 in `liquibase-update-to-latest-2.1.x.xml`, and the 2 `FOREIGN_KEY_CHECKS` toggles in `liquibase-core-data-1.9.x.xml`. Every raw-SQL changeset among these was audited individually for MariaDB compatibility (multi-table `UPDATE`/`DELETE ... JOIN`, backslash string escapes, double-quoted literals, boolean literals — all identical semantics under MariaDB's default `sql_mode`), and the three `customChange` classes involved (`BooleanConceptChangeSet`, `AddConceptMapTypesChangeset`, `MigrateConceptReferenceTermChangeSet`) contain only database-agnostic JDBC.

2. **Three changesets are deliberately left untouched** — `201610042145-2.1`, `20100128-1`, `20090831-1040-scheduler_task_config` — because extending them requires new `modifySql` blocks, which are part of the checksum. Dedicated mariadb-only changesets (`TRUNK-6634-2026-07-03-1000/-1001/-1002`) perform the equivalent work using `valueComputed="UUID()"` (no placeholder trick needed for a single dbms), each ordered before its dependent constraint. Because they filter on `uuid is null`/role-missing, they also **repair** MariaDB 11 databases already stuck at a failed constraint, and are no-ops everywhere else.

3. **The 57 uuid backfills gated by mysql/oracle `<dbms>` preconditions are intentionally not modified**: on MariaDB they are covered by the existing Java fallback in changeset `20090402-1517` (`GenerateUuid`), which already special-cases `mariadb` in its `execute()`.

New tests:

- `ChangeSetChecksumBaselineTest` pins the v8 checksums of **all 718 changesets** in the three touched changelogs to a baseline generated from the changelogs *before* this change — i.e. the values existing production databases have stored in `DATABASECHANGELOG`. All 718 were verified byte-identical after the edits, and any future edit that would break `liquibase validate` for existing deployments now fails CI. The test also enforces that mysql changesets include mariadb unless documented as mysql-only by design, and asserts the ordering of the repair changesets.
- `MariaDbUpdateChangeSetsTest` covers the four 2.1.x changesets named in the ticket in detail.

Verification notes: the full test suite passes locally except two pre-existing issues that reproduce **identically on unmodified master** in the same environment and are unrelated to this change: (a) `OpenmrsConfigurationFactoryTest#getConfigurationFiles_shouldIgnoreUnreadableFiles` — the build sandbox runs as root, so `File.setReadable(false)` cannot make a file unreadable; passes on unprivileged runners; (b) the `ModuleConfigDTDTest_*` parameterized cases for DTD versions 1.6/1.7/2.0 — `ConfigXmlBuilder` emits `loadIfModulesPresent`/`openmrsModule` elements that none of the shipped `config-1.6/1.7/2.0.dtd` files declare (nor do the copies on resources.openmrs.org), so those cases fail pure-local DTD validation as hardened in TRUNK-6441; happy to file a separate ticket for that if useful. (`ObsServiceTest#updateObs_shouldUpdateAComplexObs` also flaked once under full-suite load in the same sandbox and passes 86/86 in isolation.) This change has not been exercised against a live MariaDB 11 server as part of this PR; per the ticket discussion, the dedicated-changeset approach completed the installation wizard on a fresh MariaDB 11 container in @ranidunethma's local testing. Suggested reviewer verification: fresh install on MariaDB 11, upgrade of a MariaDB 10→11 database, and upgrade of a MySQL 8 database as the no-regression case (all 718 pinned checksums say MySQL validation is untouched, and the three new changesets are `dbms="mariadb"` only).

Relationship to #6109: complementary. That PR takes the in-place approach including `modifySql` edits; the measurements above show which parts of that are checksum-safe and which are not. Happy for maintainers to take either PR — if #6109 is preferred, the checksum baseline test here is still worth adopting on its own.

## Issue I worked on

see https://issues.openmrs.org/browse/TRUNK-6634

## Checklist: I completed these to help reviewers :)

- [x] My IDE is configured to follow the code style of this project.

- [x] I have added tests to cover my changes. (If you refactored
  existing code that was well tested you do not have to add tests)

- [x] I ran `mvn clean package` right before creating this pull request and
  added all formatting changes to my commit.

- [x] All new and existing tests passed. (See verification notes above: one pre-existing,
  environment-specific failure unrelated to this change, reproduced identically on unmodified master.)

- [x] My pull request is based on the latest changes of the master branch.
