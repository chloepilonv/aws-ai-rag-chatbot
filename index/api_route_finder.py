"""
Hierarchy-based route discovery for doc.tasq.qarnot.com documentation.

This module discovers all documentation routes by traversing the doc_hierarchy.json
file which contains the proper hierarchy structure for the documentation site.

The hierarchy file defines:
- root.base_url: Base URL for all relative paths
- root.children: Recursive tree structure with sections, groups, and pages
- root.other_doc_roots: Absolute URLs to other documentation sites (CLI, SDKs)

Node types:
- "root": Container node (not crawled)
- "section": Top-level doc area (e.g., "Getting Started", "How-to guides")
- "group": Logical grouping under a section (e.g., "Using SDKs")
- "page": Actual content pages that should be crawled

Usage:
    python index/api_route_finder.py
"""
import json
import os
from typing import List, Dict, Any


def load_hierarchy_file(file_path: str = "index/doc_hierarchy.json") -> Dict[str, Any]:
    """
    Load the documentation hierarchy from JSON file.

    Args:
        file_path: Path to the hierarchy JSON file (relative to project root)

    Returns:
        Hierarchy dictionary
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    full_path = os.path.join(project_root, file_path)

    print(f"[hierarchy] Loading hierarchy from {full_path}")

    if not os.path.exists(full_path):
        print(f"[hierarchy] ❌ Hierarchy file not found: {full_path}")
        return {}

    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[hierarchy] ✅ Loaded hierarchy file")
    return data


def extract_routes_from_hierarchy(root_data: Dict[str, Any]) -> List[str]:
    """
    Extract all page URLs from the hierarchy structure.

    This function recursively traverses the hierarchy tree to build full URLs
    for all pages in the documentation.

    Args:
        root_data: The root object from doc_hierarchy.json

    Returns:
        List of full URLs to documentation pages
    """
    routes = []
    base_url = root_data.get("base_url", "")

    print(f"[hierarchy] Base URL: {base_url}")

    def traverse(node: Dict[str, Any]) -> None:
        """Recursively traverse the hierarchy and collect page URLs."""
        node_type = node.get("type", "")
        node_url = node.get("url", "")
        node_title = node.get("title", "")

        # Only collect URLs for pages (and sections/groups that have URLs)
        if node_type == "page" and node_url:
            full_url = base_url + node_url
            routes.append(full_url)
            print(f"[hierarchy]   → {node_type}: {full_url}")
        elif node_type in ["section", "group"] and node_url:
            # Sections and groups might also be pages
            full_url = base_url + node_url
            routes.append(full_url)
            print(f"[hierarchy]   → {node_type}: {full_url}")

        # Also handle alias URLs (e.g., for Home page)
        alias_urls = node.get("alias_urls", [])
        for alias_url in alias_urls:
            full_url = base_url + alias_url
            routes.append(full_url)
            print(f"[hierarchy]   → alias: {full_url}")

        # Recursively process children
        children = node.get("children", [])
        for child in children:
            traverse(child)

    # Traverse all children of root
    root_children = root_data.get("children", [])
    for child in root_children:
        traverse(child)

    # Also add other_doc_roots (these are absolute URLs)
    other_roots = root_data.get("other_doc_roots", [])
    for other_root in other_roots:
        other_url = other_root.get("url", "")
        if other_url:
            routes.append(other_url)
            print(f"[hierarchy]   → other_doc: {other_url}")

    print(f"[hierarchy] ✅ Extracted {len(routes)} URLs from hierarchy")
    return routes


def discover_documentation_routes() -> List[str]:
    """
    Discover all documentation routes from the hierarchy JSON file.

    This function:
    1. Loads the doc_hierarchy.json file
    2. Traverses the hierarchy tree recursively
    3. Collects all page URLs (sections, groups, pages)
    4. Includes other_doc_roots (CLI, SDK docs)

    Returns:
        List of full URLs to documentation pages
    """
    print("[hierarchy] Starting hierarchy-based route discovery...")

    # Load the hierarchy file
    hierarchy_data = load_hierarchy_file()

    if not hierarchy_data:
        print("[hierarchy] ❌ No hierarchy data available")
        return []

    # Get the root object
    root = hierarchy_data.get("root", {})

    if not root:
        print("[hierarchy] ❌ No root object in hierarchy data")
        return []

    # Extract all routes from the hierarchy
    routes = extract_routes_from_hierarchy(root)

    print(f"[hierarchy] ✅ Discovered {len(routes)} documentation URLs")
    return routes


def get_documentation_urls() -> List[str]:
    """
    Get list of public documentation URLs.

    Returns:
        List of public URLs to documentation pages
    """
    return discover_documentation_routes()


def save_routes(urls: List[str], output_file: str = "index/tmp/discovered_routes.json") -> str:
    """
    Save discovered routes to a JSON file.

    Args:
        urls: List of URLs
        output_file: Path to output JSON file (relative to project root)

    Returns:
        Full path to the saved file
    """
    # Save to index/tmp/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, output_file)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        "total": len(urls),
        "routes": sorted(urls),
        "categories": {
            "getting-started": len([r for r in urls if '/getting-started/' in r]),
            "choosing-your-software": len([r for r in urls if '/choosing-your-software/' in r or '/choosing-software/' in r]),
            "core-concepts": len([r for r in urls if '/core-concepts/' in r]),
            "how-to": len([r for r in urls if '/how-to/' in r]),
            "monitoring-debugging": len([r for r in urls if '/monitoring-debugging/' in r or '/monitoring/' in r]),
            "developer-tools": len([r for r in urls if '/developer-tools/' in r]),
            "sdk-docs": len([r for r in urls if '/sdk-' in r]),
            "cli-docs": len([r for r in urls if '/cli/' in r]),
            "other": len([r for r in urls if not any(cat in r for cat in [
                '/getting-started/', '/choosing-your-software/', '/choosing-software/',
                '/core-concepts/', '/how-to/', '/monitoring-debugging/', '/monitoring/',
                '/developer-tools/', '/sdk-', '/cli/'
            ])])
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[hierarchy] 💾 Saved {len(urls)} routes to {output_path}")
    return output_path


def load_discovered_routes(file_path: str = "index/tmp/discovered_routes.json") -> List[str]:
    """
    Load previously discovered routes from JSON file.

    Args:
        file_path: Path to the JSON file (relative to project root)

    Returns:
        List of URLs
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    full_path = os.path.join(project_root, file_path)

    if not os.path.exists(full_path):
        return []

    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('routes', [])


def print_route_summary(urls: List[str]) -> None:
    """Print a summary of discovered routes by category."""
    categories = {
        "Getting Started": [r for r in urls if '/getting-started/' in r],
        "Choosing Software": [r for r in urls if '/choosing-your-software/' in r or '/choosing-software/' in r],
        "Core Concepts": [r for r in urls if '/core-concepts/' in r],
        "How-To Guides": [r for r in urls if '/how-to/' in r],
        "Monitoring & Debugging": [r for r in urls if '/monitoring-debugging/' in r or '/monitoring/' in r],
        "Developer Tools": [r for r in urls if '/developer-tools/' in r],
        "SDK Docs": [r for r in urls if '/sdk-' in r],
        "CLI Docs": [r for r in urls if '/cli/' in r],
    }

    print("\n[hierarchy] 📊 Route Summary:")
    for category, cat_routes in categories.items():
        if cat_routes:
            print(f"  {category}: {len(cat_routes)} routes")

    # Show rclone route specifically
    rclone_routes = [r for r in urls if 'rclone' in r.lower()]
    if rclone_routes:
        print(f"\n[hierarchy] 📦 Rclone route:")
        for route in rclone_routes:
            print(f"    {route}")

    # Show Docker-specific routes
    docker_routes = [r for r in urls if '/docker' in r.lower()]
    if docker_routes:
        print(f"\n[hierarchy] 🐳 Docker-related routes ({len(docker_routes)}):")
        for route in sorted(docker_routes)[:5]:  # Show first 5
            print(f"    {route}")
        if len(docker_routes) > 5:
            print(f"    ... and {len(docker_routes) - 5} more")


if __name__ == "__main__":
    # Discover routes from hierarchy file
    urls = discover_documentation_routes()

    if urls:
        # Save to file
        output_path = save_routes(urls)

        # Print summary
        print_route_summary(urls)

        print(f"\n[hierarchy] ✅ Done! Routes saved to {output_path}")
        print(f"[hierarchy] 💡 Use these routes in index_builder.py for complete documentation coverage")
    else:
        print("[hierarchy] ❌ No routes discovered")
