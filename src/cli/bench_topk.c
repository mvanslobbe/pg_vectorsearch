/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench topk
 *
 * Benchmark the candidate collection a cluster scan feeds: dedup'ing
 * inserts and the capped, sorted extract. Candidates mimic a scan of
 * posting lists: ids are posting-encoded TIDs, a share of them repeat
 * (SOAR replicas) and distances drift upward with cluster rank.
 */

#include "vs_config.h"

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "algo/topk.h"
#include "cmd.h"
#include "core/memory.h"

#define DEFAULT_ENTRIES	   40000
#define DEFAULT_K		   1000
#define DEFAULT_RUNS	   50
#define DEFAULT_ROWS_PAGE  2 /* 768-dim float vectors fill an 8 kB page */
#define DEFAULT_DUP_PCT	   30
#define POOL_PER_LIMIT_X10 64 /* rerank pool is about 6.4x the LIMIT */

typedef struct
{
	Distance distance;
	Distance error;
	uint64_t id;
} Candidate;

typedef struct
{
	uint32_t	k;
	uint32_t	cap;
	uint32_t	entries;
	uint32_t	runs;
	uint32_t	dup_pct;
	uint32_t	rows_page;
	bool		sweep;
	bool		help;
} BenchConfig;

static inline uint64_t
get_time_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Cluster-ordered stream: later clusters sit farther from the query. */
static Candidate *
generate_candidates(
		uint32_t n, uint32_t dup_pct, uint32_t rows_page, uint32_t seed)
{
	Candidate *c	 = vs_alloc(n * sizeof(Candidate));
	uint32_t   state = seed;

	for (uint32_t i = 0; i < n; i++)
	{
		float	 trend = (float)i / (float)n;
		float	 noise = (float)(cmd_rand(&state) >> 8) / 16777216.0f;
		uint64_t id;

		if (i > 0 && cmd_rand(&state) % 100 < dup_pct)
			id = c[cmd_rand(&state) % i].id;
		else
			id = ((uint64_t)(1000 + i / rows_page) << 16) |
				 (uint64_t)(1 + i % rows_page);

		c[i].distance = 0.3f + 0.4f * trend + 0.3f * noise;
		c[i].error	  = 0.04f;
		c[i].id		  = id;
	}
	return c;
}

typedef struct
{
	double insert_ms;
	double extract_ms;
	uint32_t candidates;
	uint32_t survivors;
	uint32_t returned;
} BenchResult;

static BenchResult
run_config(const BenchConfig *cfg, const Candidate *cands)
{
	BenchResult best = {1e30, 1e30, 0, 0, 0};
	VsTopK		topk;

	vs_topk_init(&topk, cfg->k);

	for (uint32_t run = 0; run < cfg->runs; run++)
	{
		vs_topk_reset(&topk);

		uint64_t t0 = get_time_ns();
		for (uint32_t i = 0; i < cfg->entries; i++)
			vs_topk_insert(
					&topk, cands[i].distance, cands[i].error, cands[i].id);
		uint64_t t1 = get_time_ns();

		VsTopKEntry *out =
				vs_alloc((topk.cand_count + 1) * sizeof(VsTopKEntry));
		uint32_t	 n	 = 0;
		uint64_t	 t2	 = get_time_ns();
		vs_topk_extract_sorted_capped(&topk, out, &n, cfg->cap);
		uint64_t t3 = get_time_ns();
		vs_free(out);

		double ins = (double)(t1 - t0) / 1e6;
		double ext = (double)(t3 - t2) / 1e6;
		if (ins < best.insert_ms)
			best.insert_ms = ins;
		if (ext < best.extract_ms)
			best.extract_ms = ext;
		best.candidates = topk.cand_count;
		best.returned	= n;
	}
	vs_topk_cleanup(&topk);
	return best;
}

static void
print_row(const char *label, const BenchConfig *cfg, const BenchResult *r)
{
	printf("%-8s %7u %7u %8u %9u %9u %10.3f %11.3f %10.3f\n",
		   label,
		   cfg->k,
		   cfg->cap,
		   cfg->entries,
		   r->candidates,
		   r->returned,
		   r->insert_ms,
		   r->extract_ms,
		   r->insert_ms + r->extract_ms);
}

static void
print_header(void)
{
	printf("%-8s %7s %7s %8s %9s %9s %10s %11s %10s\n",
		   "limit",
		   "k",
		   "cap",
		   "entries",
		   "cands",
		   "returned",
		   "insert_ms",
		   "extract_ms",
		   "total_ms");
}

static void
print_usage(const CmdContext *ctx)
{
	printf("Usage: %s bench topk [options]\n\n", ctx->prog_name);
	printf("Options:\n");
	printf("  --k N          Heap size (default %d)\n", DEFAULT_K);
	printf("  --cap N        Extract cap, 0 = none (default: k)\n");
	printf("  --entries N    Candidates scanned (default %d)\n",
		   DEFAULT_ENTRIES);
	printf("  --dup-pct N    Percent repeated ids (default %d)\n",
		   DEFAULT_DUP_PCT);
	printf("  --rows-per-page N  TID offsets per heap page (default %d)\n",
		   DEFAULT_ROWS_PAGE);
	printf("  --runs N       Repetitions, best reported (default %d)\n",
		   DEFAULT_RUNS);
	printf("  --sweep        LIMIT 10/100/1000/5000, pool-sized k and cap\n");
	printf("  --help         Show this help\n");
}

int
cmd_bench_topk(CmdContext *ctx)
{
	BenchConfig cfg = {
			.k		 = DEFAULT_K,
			.cap	 = 0,
			.entries = DEFAULT_ENTRIES,
			.runs	 = DEFAULT_RUNS,
			.dup_pct   = DEFAULT_DUP_PCT,
			.rows_page = DEFAULT_ROWS_PAGE,
	};
	bool cap_set = false;

	static struct option long_options[] = {
			{"k", required_argument, 0, 'k'},
			{"cap", required_argument, 0, 'c'},
			{"entries", required_argument, 0, 'e'},
			{"dup-pct", required_argument, 0, 'd'},
			{"rows-per-page", required_argument, 0, 'p'},
			{"runs", required_argument, 0, 'r'},
			{"sweep", no_argument, 0, 's'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0}};

	optind = 1;
	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"k:c:e:d:p:r:sh",
					long_options,
					NULL)) != -1)
	{
		switch (opt)
		{
		case 'k':
			cfg.k = (uint32_t)atoi(optarg);
			break;
		case 'c':
			cfg.cap	 = (uint32_t)atoi(optarg);
			cap_set	 = true;
			break;
		case 'e':
			cfg.entries = (uint32_t)atoi(optarg);
			break;
		case 'd':
			cfg.dup_pct = (uint32_t)atoi(optarg);
			break;
		case 'p':
			cfg.rows_page = (uint32_t)atoi(optarg);
			break;
		case 'r':
			cfg.runs = (uint32_t)atoi(optarg);
			break;
		case 's':
			cfg.sweep = true;
			break;
		case 'h':
			cfg.help = true;
			break;
		default:
			print_usage(ctx);
			return 1;
		}
	}

	if (cfg.help)
	{
		print_usage(ctx);
		return 0;
	}
	if (cfg.k == 0 || cfg.entries == 0 || cfg.runs == 0 || cfg.dup_pct > 100 ||
		cfg.rows_page == 0)
	{
		fprintf(stderr, "Error: invalid option value\n");
		return 1;
	}
	if (!cap_set)
		cfg.cap = cfg.k;

	Candidate *cands = generate_candidates(
			cfg.entries, cfg.dup_pct, cfg.rows_page, 42);

	print_header();
	if (cfg.sweep)
	{
		static const uint32_t limits[] = {10, 100, 1000, 5000};
		for (size_t i = 0; i < sizeof(limits) / sizeof(limits[0]); i++)
		{
			BenchConfig c = cfg;
			char		label[16];
			c.cap = c.k = limits[i] * POOL_PER_LIMIT_X10 / 10;
			snprintf(label, sizeof(label), "%u", limits[i]);
			BenchResult r = run_config(&c, cands);
			print_row(label, &c, &r);
		}
	}
	else
	{
		BenchResult r = run_config(&cfg, cands);
		print_row("-", &cfg, &r);
	}

	vs_free(cands);
	return 0;
}
