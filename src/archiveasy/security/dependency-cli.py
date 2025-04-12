#!/usr/bin/env python3
"""
Dependency security checker for ArchiveAsyLLM.

This script analyzes project dependencies to identify potential security issues
like slopsquatting (typosquatting targeted at LLM-generated code).
"""
# crumb: security\dependency_checker.py
import sys
import os
import argparse
import logging
from typing import Dict, List, Optional
from pathlib import Path

from archiveasy.security.validator import create_validator
from archiveasy.security.analyzer import analyze_project_dependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check dependencies for security issues')
    
    parser.add_argument('path', help='Path to the project directory')
    
    parser.add_argument(
        '--mode',
        choices=['whitelist', 'verify'],
        default='verify',
        help='Validation mode - whitelist (strict) or verify (check against PyPI)'
    )
    
    parser.add_argument(
        '--whitelist',
        help='Path to whitelist JSON file (optional)'
    )
    
    parser.add_argument(
        '--report',
        help='Path to save the analysis report (optional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate path
    project_path = args.path
    if not os.path.isdir(project_path):
        logger.error(f"Project path is not a directory: {project_path}")
        return 1
    
    # Run analysis
    try:
        logger.info(f"Analyzing project dependencies in {project_path}...")
        logger.info(f"Mode: {args.mode}")
        
        results = analyze_project_dependencies(
            project_path=project_path,
            mode=args.mode,
            whitelist_path=args.whitelist,
            save_report=True,
            report_path=args.report
        )
        
        # Print summary
        invalid_packages = results["packages"]["invalid"]
        if invalid_packages:
            logger.warning(f"Found {len(invalid_packages)} potentially unsafe packages:")
            for pkg in invalid_packages:
                logger.warning(f"  - {pkg['name']} ({pkg['type']}): {pkg['error']}")
            
            # Exit with error code if invalid packages found
            return 1
        else:
            logger.info(f"No unsafe packages found. Project dependencies look good!")
            logger.info(f"Found {len(results['packages']['imports'])} imported packages and "
                       f"{len(results['packages']['requirements'])} requirements.")
            return 0
            
    except Exception as e:
        logger.error(f"Error analyzing dependencies: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
