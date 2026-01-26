# Example 09: JSONPath Queries
#
# JSONPath is a query language for JSON, similar to XPath for XML.
# Use it to extract data from complex JSON structures.

from src import loads, jsonpath_query, jsonpath_one


fn main() raises:
    print("JSONPath Query Examples")
    print("=" * 50)
    print()
    
    # Sample data
    var data = loads('''
    {
        "store": {
            "name": "TechMart",
            "books": [
                {"title": "Mojo Programming", "price": 29.99, "category": "tech"},
                {"title": "GPU Computing", "price": 49.99, "category": "tech"},
                {"title": "Design Patterns", "price": 39.99, "category": "tech"},
                {"title": "The Great Novel", "price": 19.99, "category": "fiction"}
            ],
            "electronics": [
                {"name": "Laptop", "price": 999.99},
                {"name": "Phone", "price": 699.99}
            ]
        },
        "users": [
            {"name": "Alice", "age": 30, "active": true},
            {"name": "Bob", "age": 25, "active": false},
            {"name": "Charlie", "age": 35, "active": true}
        ]
    }
    ''')
    
    # ==========================================================
    # 1. Basic path access
    # ==========================================================
    print("1. Basic path access:")
    
    var store_name = jsonpath_query(data, "$.store.name")
    print("   $.store.name ->", store_name[0].string_value())
    print()
    
    # ==========================================================
    # 2. Array access
    # ==========================================================
    print("2. Array access:")
    
    var first_book = jsonpath_query(data, "$.store.books[0]")
    print("   $.store.books[0] ->", first_book[0]["title"].string_value())
    
    var last_book = jsonpath_query(data, "$.store.books[-1]")
    print("   $.store.books[-1] ->", last_book[0]["title"].string_value())
    print()
    
    # ==========================================================
    # 3. Wildcard - get all items
    # ==========================================================
    print("3. Wildcard (all items):")
    
    var all_titles = jsonpath_query(data, "$.store.books[*].title")
    print("   $.store.books[*].title ->")
    for i in range(len(all_titles)):
        print("     -", all_titles[i].string_value())
    print()
    
    # ==========================================================
    # 4. Recursive descent (..)
    # ==========================================================
    print("4. Recursive descent (find anywhere):")
    
    var all_names = jsonpath_query(data, "$..name")
    print("   $..name -> found", len(all_names), "names")
    for i in range(len(all_names)):
        print("     -", all_names[i].string_value())
    print()
    
    # ==========================================================
    # 5. Filter expressions
    # ==========================================================
    print("5. Filter expressions:")
    
    # Filter by value
    var tech_books = jsonpath_query(data, '$.store.books[?@.category=="tech"]')
    print("   Books where category=='tech':", len(tech_books), "found")
    
    # Numeric comparison
    var expensive = jsonpath_query(data, "$.store.books[?@.price>30]")
    print("   Books where price>30:", len(expensive), "found")
    for i in range(len(expensive)):
        print("     -", expensive[i]["title"].string_value(), "$" + String(expensive[i]["price"].float_value()))
    
    # Filter users
    var active_users = jsonpath_query(data, "$.users[?@.active==true]")
    print("   Active users:", len(active_users), "found")
    print()
    
    # ==========================================================
    # 6. Array slices
    # ==========================================================
    print("6. Array slices:")
    
    var first_two = jsonpath_query(data, "$.store.books[0:2]")
    print("   $.store.books[0:2] ->", len(first_two), "books")
    
    var every_other = jsonpath_query(data, "$.store.books[::2]")
    print("   $.store.books[::2] ->", len(every_other), "books (every other)")
    print()
    
    # ==========================================================
    # 7. Get single value with jsonpath_one
    # ==========================================================
    print("7. Get single value:")
    
    var one = jsonpath_one(data, "$.store.name")
    print("   Store name:", one.string_value())
    print()
    
    # ==========================================================
    # 8. Practical example - data extraction
    # ==========================================================
    print("8. Practical example:")
    print("   Getting all prices from the store...")
    
    var all_prices = jsonpath_query(data, "$..price")
    var total: Float64 = 0.0
    for i in range(len(all_prices)):
        total += all_prices[i].float_value()
    print("   Total value: $" + String(total))
    print()
    
    print("Done!")
