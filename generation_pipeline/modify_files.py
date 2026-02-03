"""
CLI tool for modifying files using LLM

Usage:
    python generation_pipeline/modify_files.py --files file1.json file2.md --context "Fix participant N values"
    python generation_pipeline/modify_files.py --files file1.json --context "Update demographics" --apply
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation_pipeline.utils.file_modifier import FileModifier


def main():
    parser = argparse.ArgumentParser(
        description="Modify files using LLM based on context"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Files to modify (one or more file paths)"
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Context/description of what changes to make"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes directly (creates .backup files). If not set, saves to .modified files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save modified files (default: create .modified versions in same directory)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "openai", "anthropic", "xai", "openrouter"],
        help="LLM provider (default: gemini)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gemini-3-flash-preview",
        help="Model to use (default depends on provider)"
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key (else from env)")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Path to original paper PDF (auto-detected from study directory if not provided)"
    )
    parser.add_argument(
        "--no-auto-pdf",
        action="store_true",
        help="Disable automatic PDF detection"
    )
    
    args = parser.parse_args()
    
    # Convert file paths
    file_paths = [Path(f) for f in args.files]
    
    # Check all files exist
    for file_path in file_paths:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
    
    # Initialize modifier
    modifier = FileModifier(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    
    try:
        # Modify files
        print(f"Modifying {len(file_paths)} file(s)...")
        print(f"Context: {args.context}")
        print()
        
        results = modifier.modify_files(
            file_paths=file_paths,
            context=args.context,
            output_dir=args.output_dir,
            apply_changes=args.apply,
            pdf_path=args.pdf,
            auto_find_pdf=not args.no_auto_pdf
        )
        
        # Print PDF info if used
        if results.get('pdf_used'):
            print(f"📄 Using PDF: {results['pdf_used']}")
            print()
        
        # Print results
        print("=" * 60)
        print("Modification Results")
        print("=" * 60)
        
        for file_path, result in results['files'].items():
            print(f"\nFile: {file_path}")
            print(f"  Status: {result['status']}")
            if result['status'] == 'applied':
                print(f"  Backup: {result['backup']}")
                print(f"  Modified: {result['modified']}")
            elif result['status'] == 'saved':
                print(f"  Original: {result['original']}")
                print(f"  Modified: {result['modified']}")
            elif result['status'] == 'no_changes':
                print(f"  Message: {result['message']}")
        
        print()
        print("=" * 60)
        if args.apply:
            print("✓ Changes applied. Original files backed up with .backup extension")
        else:
            print("✓ Changes saved to new files. Review before applying.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

