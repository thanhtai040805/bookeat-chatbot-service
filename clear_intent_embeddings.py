#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to clear old intent embeddings from Vector DB
"""
import asyncio
import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.vector_service import vector_service

async def clear_intent_embeddings():
    """Clear all old intent embeddings"""
    try:
        print("Clearing old intent embeddings...")
        
        # Clear all intent embeddings
        success = await vector_service.clear_all_intent_embeddings()
        
        if success:
            print("Successfully cleared all old intent embeddings")
            
            # Check stats
            stats = await vector_service.get_vector_stats()
            print(f"Intent collection now has: {stats['collections']['intents']} items")
        else:
            print("Error clearing intent embeddings")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(clear_intent_embeddings())
