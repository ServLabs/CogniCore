# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial open-source release
- 9 memory types (ABM, WM, PM, EM, AM, MM, SFM, LFM, Meta)
- Memory Management Layer (MML) with 10 learning algorithms
- Response pipeline with gate, thinking, decision, synthesis stages
- Control systems (CEN, DMN, Salience Network, Governor)
- Interfaces: admin, service, chat (WebSocket), scheduled tasks
- Observability: analytics (DuckDB) and evaluation system
- Connectors: LLM, embeddings, NLI, data, sandbox
- Comprehensive audit logging

### Changed
- Consolidated project structure (14 folders to 9)
- Moved audit, errors, migrations into core/
- Merged analytics and evals into observability/
- Reorganized APIs into interfaces/ with 4 sections

## [0.1.0] - 2024-XX-XX

### Added
- Initial release

---

## Release Notes Format

### Added
New features

### Changed
Changes in existing functionality

### Deprecated
Soon-to-be removed features

### Removed
Removed features

### Fixed
Bug fixes

### Security
Vulnerability fixes
