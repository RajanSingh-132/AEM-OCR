"""Create MongoDB databases and collections listed below (MONGO_URI from .env).

Fill in DATABASES, then run from the project folder:

    python Script_mongo/create_dbs.py --dry-run    # only show what would be created
    python Script_mongo/create_dbs.py              # create them

Safe to run again: databases / collections that already exist are left as they are
(no data is changed or deleted). MongoDB has no empty databases, so a database appears
once its first collection is created.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid, PyMongoError

# ---------------------------------------------------------------------------
# FILL IN: database name -> its collections
# ---------------------------------------------------------------------------
DATABASES: dict[str, list[str]] = {
    "2014": ["emanifest", "trips"],
    "AFMAI_Orders": ["orders"],
    "AFMAI_Invoices": ["invoices"],
}
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the MongoDB databases and collections listed in DATABASES.")
    parser.add_argument("--dry-run", action="store_true", help="only show what would be created")
    args = parser.parse_args()

    if not DATABASES:
        print("DATABASES is empty: fill in the database and collection names at the top of this file.")
        return 1
    for db_name, collections in DATABASES.items():
        if not db_name.strip() or not collections or any(not c.strip() for c in collections):
            print(f"Invalid entry {db_name!r}: {collections!r} (names must not be empty, each database needs a collection)")
            return 1

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        print("MONGO_URI is missing in .env")
        return 1

    try:
        client = MongoClient(uri, appname="AEM-AI-create-dbs", serverSelectionTimeoutMS=10_000)
        client.admin.command("ping")
    except PyMongoError as exc:
        print(f"Could not connect to MongoDB: {exc}")
        return 1

    created = existing = failed = 0
    for db_name, collections in DATABASES.items():
        db = client[db_name]
        have = set(db.list_collection_names())
        print(f"\nDatabase '{db_name}'{' (exists)' if have else ' (new)'}")
        for name in collections:
            if name in have:
                print(f"  = {name}  already exists")
                existing += 1
                continue
            if args.dry_run:
                print(f"  + {name}  would be created")
                created += 1
                continue
            try:
                db.create_collection(name)
                print(f"  + {name}  created")
                created += 1
            except CollectionInvalid:  # created by someone else meanwhile
                print(f"  = {name}  already exists")
                existing += 1
            except PyMongoError as exc:
                print(f"  ! {name}  FAILED: {exc}")
                failed += 1

    action = "would be created" if args.dry_run else "created"
    print(f"\nDone: {created} {action}, {existing} already existed, {failed} failed.")
    client.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
