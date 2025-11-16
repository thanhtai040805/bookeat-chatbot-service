"""
Script để tag tất cả menu items với LLM (offline background job)

Usage:
    python tag_menus.py                           # Tag tất cả menus (từ Spring API)
    python tag_menus.py --from-vector-db          # Tag tất cả menus (từ Vector DB)
    python tag_menus.py --restaurant 1            # Tag menu của restaurant ID 1 (từ Spring API)
    python tag_menus.py --restaurant 1 --from-vector-db  # Tag menu của restaurant ID 1 (từ Vector DB)
"""
import asyncio
import logging
import sys
import argparse
from app.services.menu_tagging_service import menu_tagging_service
from app.services.vector_service import vector_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description='Tag menu items with LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tag all menus from Spring API
  python tag_menus.py
  
  # Tag all menus from Vector DB (if Spring API not available)
  python tag_menus.py --from-vector-db
  
  # Tag specific restaurant from Spring API
  python tag_menus.py --restaurant 1
  
  # Tag specific restaurant from Vector DB
  python tag_menus.py --restaurant 1 --from-vector-db
        """
    )
    parser.add_argument(
        '--restaurant',
        type=int,
        help='Restaurant ID to tag (if not provided, tag all restaurants)'
    )
    parser.add_argument(
        '--from-vector-db',
        action='store_true',
        help='Get menus from Vector DB instead of Spring API'
    )
    args = parser.parse_args()
    
    try:
        if args.restaurant:
            logger.info(f"Starting tagging for restaurant {args.restaurant}...")
            source = "Vector DB" if args.from_vector_db else "Spring API"
            logger.info(f"Source: {source}")
            
            tagged_count = await menu_tagging_service.tag_all_menus_for_restaurant(
                args.restaurant,
                from_vector_db=args.from_vector_db
            )
            logger.info(f"✅ Completed: {tagged_count} items tagged")
        else:
            logger.info("Starting tagging job for all restaurants...")
            source = "Vector DB" if args.from_vector_db else "Spring API"
            logger.info(f"Source: {source}")
            
            stats = await menu_tagging_service.tag_all_menus(from_vector_db=args.from_vector_db)
            logger.info(f"✅ Completed: {stats['total_tagged']} items tagged across {stats['total_restaurants']} restaurants")
            
    except KeyboardInterrupt:
        logger.info("Tagging job interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in tagging job: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup QdrantClient properly để tránh ImportError khi shutdown
        try:
            if hasattr(vector_service, 'client') and vector_service.client:
                # Close Qdrant client properly
                vector_service.client.close()
        except Exception as e:
            # Ignore cleanup errors
            pass


if __name__ == "__main__":
    asyncio.run(main())

