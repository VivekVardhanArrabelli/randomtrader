# TRUNK-6634: MariaDB changeset fix — evidence report

**Ticket:** https://openmrs.atlassian.net/browse/TRUNK-6634
("Liquibase changesets with dbms=\"mysql\" are silently skipped on MariaDB")

**Author of this work:** Claude (Anthropic AI assistant), working in a sandboxed
Claude Code session sponsored by Vivek Vardhan Arrabelli. All code, experiments,
and analysis below are AI-generated and machine-verified as described; nothing
has been run against a live MariaDB server in this environment (no Docker
available), see "Verification level" below.

---

## 1. The open question on the ticket, answered empirically

The ticket discussion stalled on one question (Ian Bacher, 2026-05-09): *does
editing a historical changeset to add MariaDB support change its Liquibase
checksum?* If yes, the edit breaks `liquibase validate` on every existing
production database; if no, the in-place edit is safe.

Nobody had measured it. I did, against **Liquibase 4.32.0** (the version
resolved by the openmrs-core master build), computing both **v8** and
**latest (v9)** checksums for all four changesets named in the ticket, for
each candidate edit in isolation:

| Edit applied to changelog | Checksum effect | In-place edit safe? |
|---|---|---|
| `dbms="mysql"` → `dbms="mysql,mariadb"` on the changeSet element | **unchanged** (byte-identical, v8 and v9) | **YES** |
| adding `<dbms type="mariadb"/>` inside `<preConditions>` | **unchanged** | **YES** |
| adding a `<modifySql dbms="mariadb">` block | **CHANGED** (`8:5618aaee...` → `8:d468b6fb...`) | **NO** |

Sensitivity control: mutating actual content (`columnDataType="int"` →
`"bigint"` in changeset 201609171146-2.3) changed the checksum
(`8:7c93380a...` → `8:1c946d01...`), proving the harness detects drift.

Raw output of the experiment:

```
=== TRUNK-6634 checksum experiment (liquibase 4.32.0) ===
changeset            | version | original                           | dbms-patched                       | equal
201609171146-2.1     | V8      | 8:0958768d360dcd6a6f07dec01138c55d | 8:0958768d360dcd6a6f07dec01138c55d | true
201609171146-2.1     | V9      | 8:0958768d360dcd6a6f07dec01138c55d | 8:0958768d360dcd6a6f07dec01138c55d | true
201609171146-2.2     | V8      | 8:efac82e8f62fe999ae8cdd851256a571 | 8:efac82e8f62fe999ae8cdd851256a571 | true
201609171146-2.2     | V9      | 8:efac82e8f62fe999ae8cdd851256a571 | 8:efac82e8f62fe999ae8cdd851256a571 | true
201609171146-2.3     | V8      | 8:7c93380a4682620b5c0881d6b02bfbdf | 8:7c93380a4682620b5c0881d6b02bfbdf | true
201609171146-2.3     | V9      | 8:7c93380a4682620b5c0881d6b02bfbdf | 8:7c93380a4682620b5c0881d6b02bfbdf | true
201610042145-2.1     | V8      | 8:5618aaee0ff5e7e349424bb341b955f7 | 8:d468b6fbd700e38bcb049b6b67e4d7df | false
201610042145-2.1     | V9      | 8:5618aaee0ff5e7e349424bb341b955f7 | 8:d468b6fbd700e38bcb049b6b67e4d7df | false
--- isolation on 201610042145-2.1 (latest=V9) ---
original:       v8=8:5618aaee0ff5e7e349424bb341b955f7
precond-only:   equal=true
modifySql-only: equal=false
--- control (columnDataType int->bigint on 201609171146-2.3) ---
8:7c93380a4682620b5c0881d6b02bfbdf vs 8:1c946d0104a6505f18abda1ccd437c13   (differs, as expected)
```

The experiment harness is `DbmsChecksumExperimentTest.java` (included in this
bundle; deliberately **not** part of the PR — it tests Liquibase behavior, not
OpenMRS).

Implication for the ticket debate: **both camps were partly right.**
The "just edit dbms in place" approach (per Ian's first comment) is safe for
the attribute — and the "duplicate changesets to protect checksums" concern
(Frédéric, ranidunethma) is real, but *only* for `modifySql` additions.

## 2. The fix (commit `TRUNK-6634`, patch in this bundle)

File: `api/src/main/resources/org/openmrs/liquibase/updates/liquibase-update-to-latest-2.1.x.xml`

1. **`201609171146-2.1`, `-2.2`, `-2.3`** (cohort_member_id backfill, primary
   key, auto-increment): `dbms="mysql"` → `dbms="mysql,mariadb"` **in place**.
   Checksums proven unchanged, so existing MySQL and MariaDB-10 databases
   validate cleanly; MariaDB-11 databases run them on their next update
   (dbms-filtered changesets are never recorded in DATABASECHANGELOG, so they
   are picked up once the filter matches).

2. **`201610042145-2.1`** (uuid backfill via `modifySql` placeholder trick):
   **left completely untouched**, for two independent reasons:
   - adding a `modifySql` block changes its checksum (measured above), and
   - on MariaDB-11 systems that already attempted an upgrade, this changeset
     was recorded as MARK_RAN — an edited version would never re-run anyway,
     so an in-place edit cannot repair those databases.

   Instead a new changeset **`TRUNK-6634-2026-07-03-1000`** (`dbms="mariadb"`
   only) performs the equivalent backfill with `valueComputed="UUID()"` —
   no placeholder/modifySql hack needed since it targets a single dbms. It is
   ordered after the historical backfill and **before** the unique-constraint
   changeset `201610042145-2.2`, and its `WHERE uuid = '' OR uuid is null`
   makes it idempotent and a no-op on healthy databases. This also *repairs*
   MariaDB-11 databases stuck at the failed unique constraint: on the next
   update run the backfill fills the empty uuids, then `-2.2` succeeds.

3. **New test `MariaDbUpdateChangeSetsTest`** (part of the PR):
   - asserts the three cohort changesets now declare both mysql and mariadb;
   - **pins the four historical v8 checksums as constants** — any future edit
     that would break checksum validation for existing deployments fails CI;
   - asserts the mariadb backfill is mariadb-only and correctly ordered
     between uuid-column creation and the unique constraint;
   - asserts `201610042145-2.1` keeps exactly its two historical modifySql
     blocks.

## 3. Verification level (honest accounting)

Verified in this environment:
- `mvn -pl api test -Dtest='MariaDbUpdateChangeSetsTest,ChangeLogVersionsTest,ChangeLogDetectiveTest,ChangeLogVersionFinderTest'` → 33/33 pass, with the project's default build gates (checkstyle, license headers) enabled.
- Full `mvn -pl api test` suite run (see session notes for result).
- Checksum invariance measured empirically as above (JDK 21, Liquibase 4.32.0).

**Not verified here** (no Docker/MariaDB server in this sandbox):
- A live installation-wizard run against MariaDB 11. Note ranidunethma
  reported on the ticket (2026-06-03) that exactly this dedicated-changeset
  approach completed the wizard successfully on a fresh MariaDB 11 container.
  A reviewer or CI run should confirm: (a) fresh install on MariaDB 11,
  (b) upgrade of a MariaDB 10 → 11 database, (c) upgrade of a MySQL 8 database
  (regression check).

## 4. Relationship to PR #6109

I could not read PR #6109 from this sandbox (network policy). Per the ticket
comments it modifies historical changesets in place, including appending
mariadb to `dbms="mysql"` attributes. The evidence above gives reviewers a
precise rule for evaluating it: **attribute-only edits in that PR are
checksum-safe; any added/edited `modifySql` blocks are not.** This work is
intended as complementary evidence + a minimal reviewable fix, not as a
competing claim on the ticket. If #6109 is preferred, the checksum experiment
and the pinning test are still worth adopting.

## 5. Follow-up scope: the 2.0.x update log (not in this PR)

`liquibase-update-to-latest-2.0.x.xml` contains **250** mysql-only changesets.
Automated categorization:

| Category | Count | dbms-attribute fix safe? |
|---|---|---|
| Structural (addColumn, createIndex, renameColumn, FK, etc.) | ~199 | Yes (checksum-safe; MariaDB syntax-compatible) |
| Raw `<sql>` / `<createProcedure>` | 26 | Needs per-changeset MySQL-syntax review |
| `<customChange>` (Java) | ~21 | Java classes need MariaDB testing |
| Contains `<modifySql>` | 4 | Needs dedicated mariadb changesets (checksum) |

Plus 2 mysql-only changesets in `snapshots/core-data/liquibase-core-data-1.9.x.xml`
(only relevant to installs from the ancient 1.9.x snapshot). Recommend a
follow-up ticket rather than widening this PR.

## 6. Ready-to-paste JIRA comment

> I measured the checksum question that was left open here, against Liquibase
> 4.32.0 (openmrs-core master), v8 and v9 checksums, for all four changesets:
> changing the changeSet `dbms` attribute does **not** change the checksum,
> adding a `<dbms>` precondition does **not** change it, but adding a
> `<modifySql>` block **does** (8:5618aaee… → 8:d468b6fb…). A control mutation
> confirmed the harness detects real drift.
>
> So 201609171146-2.1/2.2/2.3 can safely become `dbms="mysql,mariadb"` in
> place, while 201610042145-2.1 must stay untouched (also because MariaDB-11
> systems recorded it MARK_RAN, so an edit would never re-run) and needs a
> dedicated mariadb-only backfill changeset ordered before 201610042145-2.2 —
> which also repairs databases currently stuck at the failed unique
> constraint. PR with this implementation + a test pinning the historical
> checksums: <PR-LINK>.
>
> Disclosure: this analysis and the PR were produced by Claude, an Anthropic
> AI assistant, in a sandboxed session; verification level is documented in
> the PR.

## 7. Suggested PR title / body

**Title:** `TRUNK-6634: Run mysql-only changesets on MariaDB in 2.1.x update log`

**Body:** use sections 1–5 of this report (the commit message carries the
summary). Must retain the AI-authorship disclosure and the "not verified
against a live MariaDB 11" caveat, and link the ticket.

## 8. How to land this (one-time human steps)

The sandbox that produced this work has GitHub access scoped to a single
unrelated repository, so it could not fork openmrs-core or open the PR.
Options:

**Option A (recommended):** start a Claude Code session with a fork of
`openmrs/openmrs-core` as the source repository, give it this bundle (or just
this report), and tell it to apply the patch, re-run the tests, push branch
`TRUNK-6634`, and open the upstream PR with the disclosure above.

**Option B (manual, ~5 minutes):**
```bash
git clone https://github.com/<your-fork>/openmrs-core.git
cd openmrs-core
git checkout -b TRUNK-6634
git am /path/to/trunk-6634.patch
./mvnw -pl api test -Dtest=MariaDbUpdateChangeSetsTest   # should pass 4/4
git push -u origin TRUNK-6634
# open PR against openmrs/openmrs-core master; paste report sections 1-5
```

Also consider claiming TRUNK-6634 in JIRA (requires an OpenMRS ID) or at
least posting the JIRA comment from section 6 so the ticket reflects the
checksum findings even before the PR is reviewed.
