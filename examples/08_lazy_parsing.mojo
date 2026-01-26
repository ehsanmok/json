# Example 08: Lazy/On-Demand Parsing
#
# Lazy parsing defers actual parsing until you access specific fields.
# Useful when you only need a few fields from a large JSON document.

from src import loads, LazyValue


fn main() raises:
    print("Lazy Parsing Examples")
    print("=" * 50)
    print()
    
    # Sample large JSON document
    var large_json = '''
    {
        "metadata": {
            "version": "2.0",
            "generated": "2024-01-15"
        },
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "profile": {"bio": "Developer"}},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "profile": {"bio": "Designer"}},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com", "profile": {"bio": "Manager"}}
        ],
        "settings": {
            "theme": "dark",
            "notifications": true,
            "language": "en"
        }
    }
    '''
    
    # ==========================================================
    # 1. Create lazy value (no parsing happens yet)
    # ==========================================================
    print("1. Create lazy value:")
    var lazy = loads[lazy=True](large_json)
    print("   Created LazyValue - no parsing yet!")
    print("   Type detected:", "object" if lazy.is_object() else "other")
    print()
    
    # ==========================================================
    # 2. Access specific paths with JSON Pointer
    # ==========================================================
    print("2. Access specific paths (only parses what's needed):")
    
    # Only parses the path to version, not the whole document
    var version = lazy.get("/metadata/version")
    print("   Version:", version.string_value())
    
    # Access nested user data
    var first_user_name = lazy.get("/users/0/name")
    print("   First user:", first_user_name.string_value())
    
    var second_user_email = lazy.get("/users/1/email")
    print("   Second user email:", second_user_email.string_value())
    
    var theme = lazy.get("/settings/theme")
    print("   Theme:", theme.string_value())
    print()
    
    # ==========================================================
    # 3. Type-specific getters
    # ==========================================================
    print("3. Type-specific getters:")
    
    var name = lazy.get_string("/users/2/name")
    print("   Third user (string):", name)
    
    var user_id = lazy.get_int("/users/0/id")
    print("   First user ID (int):", user_id)
    
    var notifications = lazy.get_bool("/settings/notifications")
    print("   Notifications (bool):", notifications)
    print()
    
    # ==========================================================
    # 4. Chain lazy access with []
    # ==========================================================
    print("4. Chain lazy access:")
    
    var users = lazy["users"]
    var first = users[0]
    var profile = first["profile"]
    var bio = profile.get("/bio")
    print("   First user's bio:", bio.string_value())
    print()
    
    # ==========================================================
    # 5. Full parse when needed
    # ==========================================================
    print("5. Full parse when needed:")
    
    var settings_lazy = lazy["settings"]
    var settings_value = settings_lazy.parse()  # Now fully parsed
    print("   Settings is object?", settings_value.is_object())
    print("   Settings keys:", len(settings_value.object_keys()))
    print()
    
    # ==========================================================
    # 6. When to use lazy parsing
    # ==========================================================
    print("6. When to use lazy parsing:")
    print("   - Large JSON documents (MB+)")
    print("   - Only need few fields from many")
    print("   - API responses where you check status first")
    print("   - Configuration files with many unused sections")
    print()
    print("   Note: Lazy parsing is CPU-only.")
    print("   For GPU speed, use loads[target='gpu'] instead.")
    print()
    
    print("Done!")
