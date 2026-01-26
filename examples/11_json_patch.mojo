# Example 11: JSON Patch (RFC 6902) and Merge Patch (RFC 7396)
#
# JSON Patch: Apply a sequence of operations to modify JSON documents.
# Merge Patch: Simpler patching by merging objects (null removes keys).

from src import loads, dumps, apply_patch, merge_patch, create_merge_patch


fn main() raises:
    print("JSON Patch Examples")
    print("=" * 50)
    print()
    
    # ==========================================================
    # 1. JSON Patch - Add operation
    # ==========================================================
    print("1. JSON Patch - Add:")
    
    var doc = loads('{"name": "Alice"}')
    var patch = loads('[{"op": "add", "path": "/age", "value": 30}]')
    
    var result = apply_patch(doc, patch)
    print("   Original:", dumps(doc))
    print("   Patch: add /age = 30")
    print("   Result:", dumps(result))
    print()
    
    # ==========================================================
    # 2. JSON Patch - Remove operation
    # ==========================================================
    print("2. JSON Patch - Remove:")
    
    doc = loads('{"name": "Alice", "age": 30, "temp": "delete me"}')
    patch = loads('[{"op": "remove", "path": "/temp"}]')
    
    result = apply_patch(doc, patch)
    print("   Original:", dumps(doc))
    print("   Patch: remove /temp")
    print("   Result:", dumps(result))
    print()
    
    # ==========================================================
    # 3. JSON Patch - Replace operation
    # ==========================================================
    print("3. JSON Patch - Replace:")
    
    doc = loads('{"name": "Alice", "status": "pending"}')
    patch = loads('[{"op": "replace", "path": "/status", "value": "active"}]')
    
    result = apply_patch(doc, patch)
    print("   Original:", dumps(doc))
    print("   Patch: replace /status = 'active'")
    print("   Result:", dumps(result))
    print()
    
    # ==========================================================
    # 4. JSON Patch - Move operation
    # ==========================================================
    print("4. JSON Patch - Move:")
    
    doc = loads('{"old_name": "value", "other": "data"}')
    patch = loads('[{"op": "move", "from": "/old_name", "path": "/new_name"}]')
    
    result = apply_patch(doc, patch)
    print("   Original:", dumps(doc))
    print("   Patch: move /old_name -> /new_name")
    print("   Result:", dumps(result))
    print()
    
    # ==========================================================
    # 5. JSON Patch - Copy operation
    # ==========================================================
    print("5. JSON Patch - Copy:")
    
    doc = loads('{"source": "original value"}')
    patch = loads('[{"op": "copy", "from": "/source", "path": "/destination"}]')
    
    result = apply_patch(doc, patch)
    print("   Original:", dumps(doc))
    print("   Patch: copy /source -> /destination")
    print("   Result:", dumps(result))
    print()
    
    # ==========================================================
    # 6. JSON Patch - Test operation
    # ==========================================================
    print("6. JSON Patch - Test (validates before applying):")
    
    doc = loads('{"version": 1, "data": "important"}')
    
    # This patch only applies if version == 1
    patch = loads('''[
        {"op": "test", "path": "/version", "value": 1},
        {"op": "replace", "path": "/version", "value": 2}
    ]''')
    
    result = apply_patch(doc, patch)
    print("   Original:", dumps(doc))
    print("   Patch: test version==1, then replace to 2")
    print("   Result:", dumps(result))
    print()
    
    # ==========================================================
    # 7. Multiple operations
    # ==========================================================
    print("7. Multiple operations:")
    
    doc = loads('{"user": {"name": "Alice", "role": "user"}}')
    patch = loads('''[
        {"op": "replace", "path": "/user/role", "value": "admin"},
        {"op": "add", "path": "/user/permissions", "value": ["read", "write"]},
        {"op": "add", "path": "/updated", "value": true}
    ]''')
    
    result = apply_patch(doc, patch)
    print("   Original:", dumps(doc))
    print("   Result:", dumps(result, indent="  "))
    print()
    
    # ==========================================================
    # 8. Merge Patch (RFC 7396) - simpler alternative
    # ==========================================================
    print("8. Merge Patch (simpler):")
    
    var target = loads('{"name": "Alice", "age": 30, "city": "NYC"}')
    var merge = loads('{"age": 31, "city": null, "country": "USA"}')
    
    result = merge_patch(target, merge)
    print("   Target:", dumps(target))
    print("   Merge:", dumps(merge))
    print("   Result:", dumps(result))
    print("   Note: null removes 'city', updates 'age', adds 'country'")
    print()
    
    # ==========================================================
    # 9. Create Merge Patch (diff two documents)
    # ==========================================================
    print("9. Create Merge Patch (diff):")
    
    var before = loads('{"name": "Alice", "age": 30}')
    var after = loads('{"name": "Alice", "age": 31, "active": true}')
    
    var diff = create_merge_patch(before, after)
    print("   Before:", dumps(before))
    print("   After:", dumps(after))
    print("   Diff patch:", dumps(diff))
    print()
    
    # ==========================================================
    # 10. Practical example - API update
    # ==========================================================
    print("10. Practical example - partial update:")
    
    var user = loads('''
    {
        "id": 123,
        "name": "Alice",
        "email": "alice@old.com",
        "settings": {"theme": "light", "notifications": true}
    }
    ''')
    
    # Client sends partial update
    var update = loads('''
    {
        "email": "alice@new.com",
        "settings": {"theme": "dark"}
    }
    ''')
    
    var updated = merge_patch(user, update)
    print("   Updated user:")
    print(dumps(updated, indent="  "))
    print()
    
    print("Done!")
