# Official SWE-bench Harness comparison

- candidate: `/mnt/data2/home/geqb/LLM/moatless-tools/experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z`
- manifest: `9b640217ff155912a70ea7ebd3cb0d5384b2819d1dfa820f21cc14f299dbe6d4`
- lock: `2a0cf8d1d84693b117c2d08fa4e40ed50b15e71f0cba0d99483fa2968358e10f`
- official source: `None` (not_provided)
- official Harness: `not_run`

| arm | rows | patches | local resolved | official resolved | local only | official only | both | official not_run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-a0 | 30 | 30 | 15 | 0 | 0 | 0 | 0 | 30 |
| dev-a1 | 30 | 30 | 16 | 0 | 0 | 0 | 0 | 30 |
| dev-a2 | 30 | 30 | 14 | 0 | 0 | 0 | 0 | 30 |
| dev-a3 | 30 | 30 | 17 | 0 | 0 | 0 | 0 | 30 |
| dev-a4 | 30 | 30 | 15 | 0 | 0 | 0 | 0 | 30 |

Official results never change the pre-registered lock or task IDs.
Environment-blocked instances with no patch stay in the denominator and are reported as missing/empty, not as resolved.
