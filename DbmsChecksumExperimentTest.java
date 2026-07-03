/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.liquibase;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import liquibase.ChecksumVersion;
import liquibase.change.CheckSum;
import liquibase.changelog.ChangeLogParameters;
import liquibase.changelog.ChangeSet;
import liquibase.changelog.DatabaseChangeLog;
import liquibase.parser.core.xml.XMLChangeLogSAXParser;
import liquibase.resource.DirectoryResourceAccessor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Scratch experiment for TRUNK-6634 — NOT intended to be merged. Empirically answers the open
 * question on the ticket: does adding "mariadb" to the dbms attribute of a changeSet, to a dbms
 * precondition, or as an additional modifySql block change the Liquibase checksum of the affected
 * changesets? The answer determines whether the fix can safely edit historical changesets in place
 * (checksums unchanged) or must duplicate them (checksums changed would break validation on
 * existing production databases).
 */
public class DbmsChecksumExperimentTest {

	private static final String CHANGELOG = "org/openmrs/liquibase/updates/liquibase-update-to-latest-2.1.x.xml";

	private static final String[] AFFECTED_IDS = { "201609171146-2.1", "201609171146-2.2", "201609171146-2.3",
	        "201610042145-2.1" };

	@TempDir
	Path tempDir;

	@Test
	public void dbmsEditsShouldNotChangeChecksums() throws Exception {
		String original = readChangelog();

		// Mutation 1: changeSet-level dbms attribute mysql -> mysql,mariadb (three changesets)
		String patched = original
		        .replace("id=\"201609171146-2.1\" dbms=\"mysql\"", "id=\"201609171146-2.1\" dbms=\"mysql,mariadb\"")
		        .replace("id=\"201609171146-2.2\" dbms=\"mysql\"", "id=\"201609171146-2.2\" dbms=\"mysql,mariadb\"")
		        .replace("id=\"201609171146-2.3\" dbms=\"mysql\"", "id=\"201609171146-2.3\" dbms=\"mysql,mariadb\"");

		// Mutation 2: add mariadb to the dbms precondition of 201610042145-2.1
		patched = patched.replace("<or><dbms type=\"mysql\"/><dbms type=\"oracle\" /></or>",
		    "<or><dbms type=\"mysql\"/><dbms type=\"mariadb\"/><dbms type=\"oracle\" /></or>");

		// Mutation 3: add a mariadb modifySql block to 201610042145-2.1
		patched = patched.replace(
		    "<modifySql dbms=\"mysql\"><replace replace=\"name-of-uuid-function\" with=\"UUID()\"/></modifySql>",
		    "<modifySql dbms=\"mysql\"><replace replace=\"name-of-uuid-function\" with=\"UUID()\"/></modifySql>\n"
		            + "\t\t<modifySql dbms=\"mariadb\"><replace replace=\"name-of-uuid-function\" with=\"UUID()\"/></modifySql>");

		// Control: a real content change that MUST alter the checksum, proving this harness is
		// sensitive enough to detect checksum drift if the dbms edits caused any.
		String control = original.replace("columnDataType=\"int\"", "columnDataType=\"bigint\"");

		assertNotEquals(original, patched, "patch was not applied");
		assertNotEquals(original, control, "control mutation was not applied");

		Map<String, CheckSum[]> originalSums = checksums(writeTemp("original.xml", original));
		Map<String, CheckSum[]> patchedSums = checksums(writeTemp("patched.xml", patched));
		Map<String, CheckSum[]> controlSums = checksums(writeTemp("control.xml", control));

		System.out.println(
		    "=== TRUNK-6634 checksum experiment (liquibase " + liquibase.util.LiquibaseUtil.getBuildVersion() + ") ===");
		System.out.printf("%-20s | %-12s | %-34s | %-34s | %s%n", "changeset", "version", "original", "dbms-patched",
		    "equal");
		for (String id : AFFECTED_IDS) {
			CheckSum[] orig = originalSums.get(id);
			CheckSum[] pat = patchedSums.get(id);
			assertNotNull(orig, "changeset " + id + " not found in original");
			assertNotNull(pat, "changeset " + id + " not found in patched");
			ChecksumVersion[] versions = { ChecksumVersion.V8, ChecksumVersion.latest() };
			for (int v = 0; v < versions.length; v++) {
				System.out.printf("%-20s | %-12s | %-34s | %-34s | %s%n", id, versions[v], orig[v], pat[v],
				    orig[v].equals(pat[v]));
			}
		}

		// Isolation: which mutation on 201610042145-2.1 moves its checksum?
		String precondOnly = original.replace("<or><dbms type=\"mysql\"/><dbms type=\"oracle\" /></or>",
		    "<or><dbms type=\"mysql\"/><dbms type=\"mariadb\"/><dbms type=\"oracle\" /></or>");
		String modifySqlOnly = original.replace(
		    "<modifySql dbms=\"mysql\"><replace replace=\"name-of-uuid-function\" with=\"UUID()\"/></modifySql>",
		    "<modifySql dbms=\"mysql\"><replace replace=\"name-of-uuid-function\" with=\"UUID()\"/></modifySql>\n"
		            + "\t\t<modifySql dbms=\"mariadb\"><replace replace=\"name-of-uuid-function\" with=\"UUID()\"/></modifySql>");
		assertNotEquals(original, precondOnly);
		assertNotEquals(original, modifySqlOnly);
		CheckSum[] precondSums = checksums(writeTemp("precond.xml", precondOnly)).get("201610042145-2.1");
		CheckSum[] modifySums = checksums(writeTemp("modifysql.xml", modifySqlOnly)).get("201610042145-2.1");
		CheckSum[] origUuid = originalSums.get("201610042145-2.1");
		System.out.println("--- isolation on 201610042145-2.1 (latest=" + ChecksumVersion.latest() + ") ---");
		System.out.printf("original:       v8=%s latest=%s%n", origUuid[0], origUuid[1]);
		System.out.printf("precond-only:   v8=%s latest=%s equal=%s%n", precondSums[0], precondSums[1],
		    origUuid[0].equals(precondSums[0]) && origUuid[1].equals(precondSums[1]));
		System.out.printf("modifySql-only: v8=%s latest=%s equal=%s%n", modifySums[0], modifySums[1],
		    origUuid[0].equals(modifySums[0]) && origUuid[1].equals(modifySums[1]));

		// Sensitivity control on 201609171146-2.3 (the addAutoIncrement changeset)
		CheckSum[] origAuto = originalSums.get("201609171146-2.3");
		CheckSum[] ctrlAuto = controlSums.get("201609171146-2.3");
		System.out.println("--- control (columnDataType int->bigint on 201609171146-2.3) ---");
		System.out.printf("v8:     %s vs %s | latest: %s vs %s%n", origAuto[0], ctrlAuto[0], origAuto[1], ctrlAuto[1]);
		assertNotEquals(origAuto[0], ctrlAuto[0], "control v8 checksum should differ — harness not sensitive!");
		assertNotEquals(origAuto[1], ctrlAuto[1], "control latest checksum should differ — harness not sensitive!");

		// The actual claim under test: dbms-related edits leave checksums untouched
		for (String id : AFFECTED_IDS) {
			assertEquals(originalSums.get(id)[0], patchedSums.get(id)[0], id + " v8 checksum changed");
			assertEquals(originalSums.get(id)[1], patchedSums.get(id)[1], id + " latest checksum changed");
		}
	}

	private String readChangelog() throws Exception {
		try (InputStream in = getClass().getClassLoader().getResourceAsStream(CHANGELOG)) {
			assertNotNull(in, "changelog not found on classpath: " + CHANGELOG);
			return new String(in.readAllBytes(), StandardCharsets.UTF_8);
		}
	}

	private String writeTemp(String name, String content) throws Exception {
		Path file = tempDir.resolve(name);
		Files.writeString(file, content, StandardCharsets.UTF_8);
		return name;
	}

	private Map<String, CheckSum[]> checksums(String fileName) throws Exception {
		DatabaseChangeLog changeLog = new XMLChangeLogSAXParser().parse(fileName, new ChangeLogParameters(),
		    new DirectoryResourceAccessor(tempDir));
		Map<String, CheckSum[]> result = new LinkedHashMap<>();
		for (ChangeSet changeSet : changeLog.getChangeSets()) {
			result.put(changeSet.getId(), new CheckSum[] { changeSet.generateCheckSum(ChecksumVersion.V8),
			        changeSet.generateCheckSum(ChecksumVersion.latest()) });
		}
		return result;
	}
}
