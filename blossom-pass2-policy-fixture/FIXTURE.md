<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->
# E06 static scan fixture

This isolated draft PR intentionally includes lodash 4.17.11 as BlackDuck scan input. Do not install or execute it. The package archive is the unmodified npm release, verified against the registry SHA512 integrity value in package-lock.json; the archive retains the upstream license. Lockfile v1 is used for compatibility with older dependency parsers.

Candidate CVE: https://github.com/advisories/GHSA-35jh-r3h4-6jhm (CVE-2021-23337, lodash <4.17.21). A published advisory does not establish a BlackDuck finding. Passing E06 requires an actual new Blossom report naming a lodash CVE, scan-failure feedback/audit, exit 255, and zero webhook requests.
