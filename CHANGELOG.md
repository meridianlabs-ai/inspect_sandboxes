# Changelog

## [0.4.1](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.4.0...v0.4.1) (2026-07-30)


### Bug Fixes

* cap Modal sandbox names at 64 characters ([#54](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/54)) ([d646c2e](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/d646c2e0fd5bd68a62491c68849255358c503d06))
* don't re-kill already-cleaned-up E2B sandboxes in task_cleanup() ([#39](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/39)) ([#47](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/47)) ([be92daf](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/be92dafbc0500ea3c72089248f94753905521de3))
* forward disk size to Daytona sandbox creation ([#50](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/50)) ([34d28ac](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/34d28aca396e64b47bb201a66808a61ac9ab9530))
* recover from failed Daytona sandbox creates ([#51](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/51)) ([4cdb6e2](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/4cdb6e2353f1f5d91ccb16a514c118cb440d3472))

## [0.4.0](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.3.0...v0.4.0) (2026-06-09)


### Features

* add E2B sandbox implementation ([#25](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/25)) ([747754a](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/747754ab581aff04b78b2804370508628851add1))
* **compose:** translate ports and surface them through connection() ([#33](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/33)) ([ac40ab2](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/ac40ab236fabb52ca390e7c63c990604918816f7))


### Bug Fixes

* **compose:** clamp positive sub-MiB memory limits to 1 MiB instead of erroring ([#29](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/29)) ([37c2bdb](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/37c2bdbaf6598605ae1ffc5929ab1b78e3814d48))
* **daytona:** support async-generator list() API (daytona-sdk &gt;=0.180) ([#34](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/34)) ([6037d4d](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/6037d4d0325ea72f74b4138c5adf308e9de45264))
* **docs:** use Sample objects in hello-world examples ([#27](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/27)) ([12e2feb](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/12e2feb16b08efaf661123a0317ab1a0ebb5fcee))
* **e2b:** root-user default, Dockerfile COPY resolution, SDK min version ([#31](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/31)) ([f57bb81](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/f57bb818c63b62a6e3e0b9dd2e0c39f2c29e7fc9))


### Documentation

* dedicated homepage with provider tabset + extension update ([#32](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/32)) ([ee7ad96](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/ee7ad963e5898116c3d3abdcaee59c1e34f0437d))

## [0.3.0](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.2.3...v0.3.0) (2026-04-24)


### Features

* quarto docs site, sandbox naming, and x-daytona.timeout fix ([#22](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/22)) ([bc0e84d](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/bc0e84d056ecda39b2a757be0a50217c92aab971))

## [0.2.3](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.2.2...v0.2.3) (2026-04-22)


### Bug Fixes

* replace inspect-ai git dep with PyPI version ([#20](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/20)) ([9722e3c](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/9722e3c0315ac51e52a438382cdb87f3ccd026a3))

## [0.2.2](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.2.1...v0.2.2) (2026-04-21)


### Bug Fixes

* **daytona:** remove env var space-splitting workaround ([#18](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/18)) ([9a18e16](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/9a18e165ca1893590ec87a9d2bdb333c60a277c6))

## [0.2.1](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.2.0...v0.2.1) (2026-04-14)


### Bug Fixes

* add DinD self-check test and document timeout cleanup behavior ([#16](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/16)) ([60dfe1e](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/60dfe1e78f01602f90b25006216233599daad3f0))

## [0.2.0](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.1.2...v0.2.0) (2026-04-03)


### Features

* **daytona:** add multi-service DinD support ([#15](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/15)) ([a1fcca6](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/a1fcca684581e51884bae26b10eeb0cf02daf9fb))
* **daytona:** clean up orphaned sandboxes on build failure ([#11](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/11)) ([30ccafc](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/30ccafca92bdec41d5959891b01c6f381b629abc))
* **modal,daytona:** add transient error retry to exec and fill retry gaps ([#14](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/14)) ([93b016d](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/93b016d737404ebca72ed5fdb4ba24caf914a131))
* **modal:** add support for modal secrets from compose files ([#13](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/13)) ([b8b1045](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/b8b10456a010c91843aa261e9ef1268227750afa))

## [0.1.2](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.1.1...v0.1.2) (2026-03-18)


### Documentation

* add network_mode notes and known limitations to READMEs ([#9](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/9)) ([e17230f](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/e17230f512b0dccbce75b6d1bdd517f3701b197b))

## [0.1.1](https://github.com/meridianlabs-ai/inspect_sandboxes/compare/v0.1.0...v0.1.1) (2026-03-16)


### Bug Fixes

* improve Modal command passing, Daytona stdin support, and network_mode translation ([#7](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/7)) ([15cbe15](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/15cbe1508c11d61a881341e997b4699999cadb72))
* pass build context to Modal image builds and skip retry on RemoteError ([1f190b3](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/1f190b33a2309cfe172fa719f7dd9d4d3087dbaf))

## 0.1.0 (2026-03-02)


### Features

* add Daytona sandbox implementation ([#2](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/2)) ([d6a9908](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/d6a9908b39606dfdba67d6e1ae96217ef6aa201a))
* add Modal sandbox implementation ([#1](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/1)) ([f3c8e73](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/f3c8e734da00f9531337de047a668c0e8b424db4))
* add PyPI publishing with release-please automation ([#3](https://github.com/meridianlabs-ai/inspect_sandboxes/issues/3)) ([3cec1bc](https://github.com/meridianlabs-ai/inspect_sandboxes/commit/3cec1bc3b64da7b458d52512a618c4187d492c8a))

## 0.1.0 (2026-03-02)

### Features

* Initial release of inspect_sandboxes
* Daytona sandbox implementation
* Modal sandbox implementation
