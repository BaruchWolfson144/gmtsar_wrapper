# InSAR Time Series Processing - Python Wrapper Project

## Project Overview

This project is developing a **Python wrapper for InSAR Time Series (SBAS) processing** based on GMTSAR and Sentinel-1 satellite data. The wrapper replaces legacy csh scripts with a modern, modular Python implementation while maintaining full compatibility with the GMTSAR library workflow.

**Primary Objective**: Transform the manual, script-based SBAS processing pipeline into a well-structured, documented, and controllable Python framework.

## Critical Reference Documentation

**MUST READ**: `sentinel_time_series.pdf` - The complete manual detailing the entire SBAS processing workflow from start to finish. This document is the authoritative source for:
- Processing steps and their order
- GMTSAR commands and parameters
- Expected inputs and outputs at each stage
- Data formats and conventions
- Troubleshooting and validation procedures

**Always refer to this manual** when implementing any processing stage or making architectural decisions.

## Technical Stack

### Core Technologies
- **Language**: Python 3.8+
- **Platform**: Linux (required for GMTSAR)
- **Primary Library**: GMTSAR (GMT-based SAR processing)
- **Data Source**: Sentinel-1 SAR satellite imagery
- **Method**: SBAS (Small Baseline Subset) time series analysis

### Key Python Libraries (Current/Planned)
- `subprocess` - Interface to GMTSAR shell commands
- `numpy` - Numerical operations
- `pandas` - Time series data management
- `pathlib` - Modern file path handling
- `logging` - Comprehensive logging framework
- `configparser` or `yaml` - Configuration management
- Consider: `click` or `argparse` for CLI

## Project Philosophy & Development Principles

### 1. Modularity
- **Each processing stage = separate module**
- Clear separation of concerns
- Reusable components
- Easy to test independently
- Simple to extend or modify individual stages

### 2. Control & Transparency
- User visibility into every processing step
- Configurable parameters at all stages
- Intermediate output preservation options
- Progress reporting and logging
- Validation checkpoints between stages

### 3. Documentation Excellence
- **Every function must have comprehensive docstrings**
- Explain the "why" not just the "what"
- Include parameter types, return values, and raised exceptions
- Document GMTSAR command equivalents
- Add usage examples in docstrings
- Maintain architectural decision records (ADR)

### 4. Professional Python Standards
- Follow PEP 8 style guide
- Type hints for all function signatures
- Descriptive variable and function names (no cryptic abbreviations)
- Error handling with specific exception types
- Unit tests for all core functionality

## Code Development Guidelines

### Function Design
When writing any function:

1. **Start with the signature**

2. **Write the docstring**
  
3. **Implement with clear structure**

## Interaction Protocol

When you receive a task or question:

1. **Analyze in context**: Consider how this fits into the overall SBAS pipeline
2. **Check the manual**: Verify against `sentinel_time_series.pdf`
3. **Present alternatives**: If multiple approaches exist:
   - List each option
   - Analyze pros/cons
   - Recommend the best fit with justification
4. **Explain architecture**: Before coding, describe:
   - Overall design approach
   - Module boundaries
   - Data flow
   - Integration points
5. **Suggest names**: Propose meaningful names for functions, classes, modules
6. **Write complete code**: Include:
   - Full docstrings
   - Type hints
   - Error handling
   - Logging statements
   - Usage examples in comments

## Key Technical Considerations

### GMTSAR Integration
- GMTSAR commands are invoked via subprocess
- Parse GMTSAR output for errors and warnings
- Maintain parameter files (.PRM) compatibility
- Handle GMTSAR's complex file dependencies

### File Management
- Track all intermediate and final outputs
- Implement cleanup strategies (temporary vs. persistent files)
- Use consistent naming conventions for outputs
- Validate file existence and format before processing

### Performance & Scalability
- Consider parallel processing opportunities (multiple interferogram pairs)
- Memory management for large raster datasets
- Progress bars for long-running operations
- Checkpoint/resume capability for interrupted processing

## Questions to Always Ask Yourself

- Does this align with the manual's processing flow?
- Is this module sufficiently decoupled from others?
- Can a user understand what this function does without reading the code?
- Are all error cases handled?
- Is the logging informative enough for debugging?
- Would this be easy to test?

## References & Resources

- GMTSAR documentation: https://topex.ucsd.edu/gmtsar/
- Sentinel-1 technical specs: ESA documentation
- SBAS methodology: Berardino et al. (2002) paper
- Project manual: `sentinel_time_series.pdf` (PRIMARY REFERENCE)

---

**Remember**: Quality over speed. This project is about creating a maintainable, professional tool that will serve as a foundation for InSAR time series analysis. Take time to design properly, document thoroughly, and implement robustly.
