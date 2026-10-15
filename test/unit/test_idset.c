/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_idset.c - Unit tests for the open-addressing id set
 *
 * Tests cover:
 * - Add/membership semantics (test_add returns newly-added)
 * - Id 0 handling via the dedicated flag
 * - Collision chains at the sizing bound (load factor <= 1/2)
 * - Structured id patterns (posting-encoded TIDs)
 * - Probe chains stay short for TIDs that share an offset
 */

#include "core/idset.h"
#include "vs_test.h"

TEST_GROUP(IdSet);
TEST_MEMCTX_FIXTURE();

TEST(idset_add_and_membership)
{
	VsIdSet set;
	vs_idset_init(&set, 8);

	ASSERT_TRUE(vs_idset_test_add(&set, 42), "first add is new");
	ASSERT_TRUE(!vs_idset_test_add(&set, 42), "second add is a member");
	ASSERT_TRUE(vs_idset_test_add(&set, 43), "distinct id is new");
	ASSERT_TRUE(!vs_idset_test_add(&set, 43), "and then a member");

	vs_idset_cleanup(&set);
}

TEST(idset_zero_id)
{
	VsIdSet set;
	vs_idset_init(&set, 4);

	ASSERT_TRUE(vs_idset_test_add(&set, 0), "id 0 is new");
	ASSERT_TRUE(!vs_idset_test_add(&set, 0), "id 0 becomes a member");
	ASSERT_TRUE(vs_idset_test_add(&set, 1), "id 1 unaffected by id 0");

	vs_idset_cleanup(&set);
}

TEST(idset_full_population)
{
	/* Fill to the sizing bound: every id distinct, then every id a
	 * duplicate, across a range wide enough to force probe chains. */
	const uint32_t n = 1000;

	VsIdSet set;
	vs_idset_init(&set, n);

	for (uint32_t i = 0; i < n; i++)
	{
		/* Posting-encoded shape: (block << 16) | offset */
		uint64_t id = ((uint64_t)(i / 3) << 16) | ((i % 3) + 1);
		/* Populating: every id here is new, and the second pass below
		 * is what reads the result. */
		(void)vs_idset_test_add(&set, id);
	}
	uint32_t dups = 0;
	for (uint32_t i = 0; i < n; i++)
	{
		uint64_t id = ((uint64_t)(i / 3) << 16) | ((i % 3) + 1);
		if (!vs_idset_test_add(&set, id))
			dups++;
	}
	ASSERT_EQ(n, dups, "every re-add reports membership");

	vs_idset_cleanup(&set);
}

TEST(idset_tid_offsets_do_not_cluster)
{
	/* Many blocks, few distinct offsets. A hash that sees only the low
	 * bits of the id sends every id with the same offset to one slot. */
	const uint32_t n = 20000;

	VsIdSet set;
	vs_idset_init(&set, n);

	for (uint32_t i = 0; i < n; i++)
	{
		uint64_t id = ((uint64_t)(100 + i / 4) << 16) | ((i % 4) + 1);
		(void)vs_idset_test_add(&set, id);
	}

	uint64_t displaced = 0;
	for (uint32_t slot = 0; slot <= set.mask; slot++)
	{
		uint64_t id = set.slots[slot];
		if (id == 0)
			continue;
		uint32_t home = vs_idset_home_slot(&set, id);
		displaced += (slot - home) & set.mask;
	}
	ASSERT_TRUE(displaced / n < 8, "probe chains stay short");

	vs_idset_cleanup(&set);
}
