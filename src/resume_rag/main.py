import argparse
import logging
import sys
from pathlib import Path
from pydantic import ValidationError
from resume_rag.app.rag_system import RAGSystem
from resume_rag.domain.models import User, Role

def setup_logging(config_manager):
    log_file = Path(config_manager.logs_dir) / "rag_system.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a')
        ]
    )

def create_test_user(user_type: str = "analyst") -> User:
    # Create a test user
    user_configs = {
        "admin": {"user_id": "admin_user", "role": Role.ADMIN},
        "hr": {"user_id": "hr_manager", "role": Role.HR_MANAGER, "department": "HR"},
        "recruiter": {"user_id": "recruiter_user", "role": Role.RECRUITER, "department": "HR"},
        "analyst": {"user_id": "analyst_user", "role": Role.ANALYST, "department": "IT"}
    }

    config = user_configs.get(user_type, user_configs["analyst"])

    try:
        return User(**config)
    except ValidationError as e:
        logging.getLogger(__name__).error(f"Failed to create test user: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Resume RAG System with Pydantic Validation")
    parser.add_argument("--csv-file", default="Resume.csv", help="CSV filename in data directory")
    parser.add_argument("--load-pdfs", action="store_true", help="Load PDF files along with CSV")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--query", help="Test query")
    parser.add_argument("--user-type", choices=["admin", "hr", "recruiter", "analyst"],
                        default="analyst", help="Type of test user")
    parser.add_argument("--config", default="config/app_config.yaml", help="Config file path")
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip indexing from CSV/PDF (use with delete/update or query-only)",
    )
    parser.add_argument(
        "--delete-docs",
        help="Comma-separated logical document IDs (metadata id) to remove from Chroma",
    )
    parser.add_argument(
        "--update-doc",
        help="Logical document ID to replace (removes all chunks for that id, then re-indexes split text)",
    )
    parser.add_argument(
        "--update-file",
        help="File path whose contents become the new document text (with --update-doc)",
    )
    parser.add_argument(
        "--update-text",
        help="Inline new document text (with --update-doc)",
    )
    parser.add_argument(
        "--update-category",
        default="Unknown",
        help="Category metadata when using --update-doc",
    )
    parser.add_argument(
        "--update-source",
        default="csv",
        help="Source metadata when using --update-doc (e.g. csv, pdf)",
    )

    args = parser.parse_args()

    if (args.query or args.evaluate) and not args.skip_load:
        print("Warning: running without --skip-load will re-index Resume.csv and can create duplicate vectors.")
        print("Tip: index once, then use --skip-load for query/evaluate workflows.")

    try:
        # Initialize system
        print("Initializing RAG System...")
        rag = RAGSystem(args.config)

        # Setup logging
        setup_logging(rag.config)
        logger = logging.getLogger(__name__)

        # Load dataset
        if args.skip_load:
            print("Skipping dataset load (--skip-load).")
            doc_count = 0
        else:
            print(f"Loading dataset from {args.csv_file}...")
            doc_count = rag.load_dataset(
                csv_filename=args.csv_file,
                load_pdfs=args.load_pdfs,
            )
            print(f"Successfully loaded {doc_count} documents")

        # Create test user
        user = create_test_user(args.user_type)
        print(f"Created test user: {user.user_id} (Role: {user.role})")

        if args.delete_docs:
            ids = [x.strip() for x in args.delete_docs.split(",") if x.strip()]
            if not ids:
                print("No valid IDs in --delete-docs")
            else:
                print(f"Deleting documents from index: {ids}")
                deleted = rag.delete_documents(ids, user=user)
                print(f"Delete {'succeeded' if deleted else 'failed or denied'}")

        if args.update_doc:
            if args.update_file and args.update_text:
                print("Use only one of --update-file or --update-text with --update-doc")
                sys.exit(1)
            if args.update_file:
                uf = Path(args.update_file)
                if not uf.is_file():
                    print(f"Update file not found: {uf}")
                    sys.exit(1)
                new_body = uf.read_text(encoding="utf-8")
            elif args.update_text:
                new_body = args.update_text
            else:
                print("--update-doc requires --update-file or --update-text")
                sys.exit(1)
            meta = {"category": args.update_category, "source": args.update_source}
            print(f"Updating document {args.update_doc!r} in index...")
            updated = rag.update_document(args.update_doc, new_body, meta, user=user)
            print(f"Update {'succeeded' if updated else 'failed or denied'}")

        # Run evaluation if requested
        if args.evaluate:
            print("\nRunning evaluation...")
            try:
                eval_results = rag.run_evaluation(user=user)
                print(f"Evaluation completed successfully with {eval_results.total_queries} queries")
            except PermissionError as e:
                print(f"Permission denied: {e}")
            except Exception as e:
                print(f"Evaluation failed: {e}")

        # Test query if provided
        if args.query:
            print(f"\nSearching for: '{args.query}'")
            print(f"User: {user.user_id} ({user.role})")

            try:
                results = rag.search(args.query, k=5, user=user)

                if results:
                    print(f"\nGenerated answer:")
                    answer = rag.generate_answer(args.query, results, user=user)
                    ans = (answer or "").strip()
                    if ans.startswith("{"):
                        try:
                            import json as _json

                            print(_json.dumps(_json.loads(ans), indent=2))
                        except _json.JSONDecodeError:
                            print(answer)
                    else:
                        print(f"{answer}\n")
                    print()

                    print(f"Top {len(results)} results:")
                    print("-" * 80)
                    for i, result in enumerate(results):
                        doc = result.document
                        print(f"{i+1}. Score: {result.score:.3f} | Method: {result.method}")
                        print(f"   Category: {doc.metadata.category}")
                        print(f"   ID: {doc.metadata.id}")
                        print(f"   Content: {doc.page_content[:200]}...")
                        print("-" * 80)
                else:
                    print("No results found for the query")

            except Exception as e:
                print(f"Search failed: {e}")

        # System statistics
        print("\nSystem Statistics:")
        stats = rag.get_system_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nRAG System execution completed successfully!")

    except ValidationError as e:
        print(f"Validation error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()