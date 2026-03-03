## 2025-03-03 - [Cache Map Keys by Identifier, Not Object ID]
**Learning:** When caching computed properties of dictionaries (like sorted keys) to avoid redundant operations, using `id(dict_obj)` as the cache key is an anti-pattern. If the dictionary is dynamically generated and later garbage collected, Python can reuse its memory address for a completely different dictionary. This leads to cache poisoning and incorrect lookups.
**Action:** Always use a stable, semantic string identifier (e.g., a league name, or a specific constant string like `"NCAAB_LEGACY"`) to key the cache instead of `id()`.

## 2025-03-03 - [Preserve Safe Type Casting in Optimization]
**Learning:** When optimizing string cleaning functions (like moving regex compilation out of the function scope), be careful not to inadvertently remove defensive type casting (like `str(name or "")`) that was present in the original implementation. Even if type hints specify `str`, runtime inputs in data pipelines can often be numbers, NaNs, or `None`.
**Action:** Compare the pre-optimization and post-optimization code line-by-line to ensure no fallback logic or defensive casting was dropped.
