"""Weaviate database connection and data loading."""

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
import joblib
import pandas as pd
import os
from typing import Optional
from config import WEAVIATE_URL, WEAVIATE_GRPC_PORT, PRODUCTS_FILE, FAQ_FILE



def get_client():
    """
    Connect to local Weaviate instance.
    
    Returns:
        weaviate.WeaviateClient: Connected Weaviate client
    """
    client = weaviate.connect_to_local(
        port=int(WEAVIATE_URL.split(':')[-1]),  # Extract port from URL
        grpc_port=WEAVIATE_GRPC_PORT
    )
    print("✅ Connected to Weaviate")
    return client


def create_products_collection(client):
    """
    Create the Products collection with proper schema.
    
    The schema includes all product metadata fields for filtering:
    - gender, masterCategory, subCategory, articleType
    - baseColour, season, year, usage
    - productDisplayName, price, product_id
    """
    # Check if collection already exists
    if client.collections.exists("products"):
        print("⚠️  Products collection already exists")
        return client.collections.get("products")
    
    # Create collection with text2vec-transformers for automatic embeddings
    products_collection = client.collections.create(
        name="products",
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(
            vectorize_collection_name=False
        ),
        properties=[
            Property(name="gender", data_type=DataType.TEXT),
            Property(name="masterCategory", data_type=DataType.TEXT),
            Property(name="subCategory", data_type=DataType.TEXT),
            Property(name="articleType", data_type=DataType.TEXT),
            Property(name="baseColour", data_type=DataType.TEXT),
            Property(name="season", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.NUMBER),
            Property(name="usage", data_type=DataType.TEXT),
            Property(name="productDisplayName", data_type=DataType.TEXT),
            Property(name="price", data_type=DataType.NUMBER),
            Property(name="product_id", data_type=DataType.NUMBER),
        ]
    )
    print("✅ Created Products collection")
    return products_collection


def create_faq_collection(client):
    """
    Create the FAQ collection with proper schema.
    
    The schema includes:
    - question: The FAQ question
    - answer: The answer to the question
    - type: Category of the FAQ (e.g., 'general information')
    """
    # Check if collection already exists
    if client.collections.exists("faq"):
        print("⚠️  FAQ collection already exists")
        return client.collections.get("faq")
    
    # Create collection with text2vec-transformers for automatic embeddings
    faq_collection = client.collections.create(
        name="faq",
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(
            vectorize_collection_name=False
        ),
        properties=[
            Property(name="question", data_type=DataType.TEXT),
            Property(name="answer", data_type=DataType.TEXT),
            Property(name="type", data_type=DataType.TEXT),
        ]
    )
    print("✅ Created FAQ collection")
    return faq_collection


def load_products(client, batch_size: int = 100, force_reload: bool = False):
    """
    Load products from joblib file into Weaviate.
    
    Args:
        client: Weaviate client
        batch_size: Number of products to insert per batch (default 100)
        force_reload: If True, delete and reload all data (default False)
    """
    # Get or create collection
    if not client.collections.exists("products"):
        products_collection = create_products_collection(client)
    else:
        products_collection = client.collections.get("products")
    
    # Check if already loaded
    current_count = len(products_collection)
    if current_count > 0:
        if not force_reload:
            print(f"⚠️  Products collection already has {current_count} items. Use force_reload=True to reload.")
            return
        else:
            print(f"🔄 Deleting {current_count} existing products and reloading...")
            client.collections.delete("products")
            products_collection = create_products_collection(client)
    
    # Load data from joblib
    print(f"📂 Loading products from {PRODUCTS_FILE}...")
    products_data = joblib.load(PRODUCTS_FILE)
    print(f"📊 Loaded {len(products_data)} products")
    
    # Insert in batches
    with products_collection.batch.dynamic() as batch:
        for i, product in enumerate(products_data):
            # Handle NaN values in year field (pandas NaN check)
            if pd.isna(product.get('year')):
                product['year'] = 0.0  # Default to 0 for missing years
            
            # Add product to batch
            batch.add_object(properties=product)
            
            # Progress indicator every 5000 items
            if (i + 1) % 5000 == 0:
                print(f"  Inserted {i + 1}/{len(products_data)} products...")
    
    print(f"✅ Loaded {len(products_data)} products into Weaviate")


def load_faqs(client, batch_size: int = 25, force_reload: bool = False):
    """
    Load FAQs from joblib file into Weaviate.
    
    Args:
        client: Weaviate client
        batch_size: Number of FAQs to insert per batch (default 25)
        force_reload: If True, delete and reload all data (default False)
    """
    # Get or create collection
    if not client.collections.exists("faq"):
        faq_collection = create_faq_collection(client)
    else:
        faq_collection = client.collections.get("faq")
    
    # Check if already loaded
    current_count = len(faq_collection)
    if current_count > 0:
        if not force_reload:
            print(f"⚠️  FAQ collection already has {current_count} items. Use force_reload=True to reload.")
            return
        else:
            print(f"🔄 Deleting {current_count} existing FAQs and reloading...")
            client.collections.delete("faq")
            faq_collection = create_faq_collection(client)
    
    # Load data from joblib
    print(f"📂 Loading FAQs from {FAQ_FILE}...")
    faq_data = joblib.load(FAQ_FILE)
    print(f"📊 Loaded {len(faq_data)} FAQs")
    
    # Insert in batches
    with faq_collection.batch.dynamic() as batch:
        for faq in faq_data:
            batch.add_object(properties=faq)
    
    print(f"✅ Loaded {len(faq_data)} FAQs into Weaviate")


def setup_database(force_reload: bool = False):
    """
    Complete database setup: connect, create collections, and load data.
    
    Args:
        force_reload: If True, delete existing data and reload (default False)
    
    This is the main function to call to initialize the database.
    """
    print("🚀 Setting up Weaviate database...")
    
    # Connect
    client = get_client()
    
    # Create collections
    create_products_collection(client)
    create_faq_collection(client)
    
    # Load data
    load_products(client, force_reload=force_reload)
    load_faqs(client, force_reload=force_reload)
    
    print("✅ Database setup complete!")
    
    return client


def reset_database(client):
    """
    Delete all collections and their data.
    WARNING: This will permanently delete all data!
    
    Args:
        client: Weaviate client
    """
    print("⚠️  Resetting database...")
    
    if client.collections.exists("products"):
        client.collections.delete("products")
        print("  Deleted Products collection")
    
    if client.collections.exists("faq"):
        client.collections.delete("faq")
        print("  Deleted FAQ collection")
    
    print("✅ Database reset complete")


if __name__ == "__main__":
    """
    Run this script directly to set up the database:
    
    Usage:
        python src/database.py              # Load only if empty
        python src/database.py --force      # Force reload all data
        python src/database.py --reset      # Delete all collections
    """
    import sys
    
    try:
        # Parse command line arguments
        force_reload = "--force" in sys.argv
        reset_mode = "--reset" in sys.argv
        
        if reset_mode:
            # Reset mode: delete everything
            print("⚠️  RESET MODE: This will delete all data!")
            confirm = input("Type 'yes' to confirm: ")
            if confirm.lower() == 'yes':
                client = get_client()
                reset_database(client)
                client.close()
            else:
                print("❌ Reset cancelled")
        else:
            # Normal setup mode
            client = setup_database(force_reload=force_reload)
            
            # Verify the data
            products = client.collections.get("products")
            faqs = client.collections.get("faq")
            
            print(f"\n📊 Database Statistics:")
            print(f"  Products: {len(products)}")
            print(f"  FAQs: {len(faqs)}")
            
            # Close connection
            client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()