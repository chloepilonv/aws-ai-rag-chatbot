"""
Automatic route discovery for Vue.js/Nuxt.js documentation sites.

This module discovers all routes from doc.tasq.qarnot.com by extracting
the navigation hierarchy from the Vue/Nuxt global state (window.__NUXT__).

Usage:
    python index/route_discovery.py

This will generate 'discovered_routes.json' containing all documentation URLs.
"""
import json
import os
from typing import List, Dict, Any
from playwright.sync_api import sync_playwright


def discover_routes_from_nuxt(base_url: str) -> List[str]:
    """
    Discover all documentation routes from a Nuxt.js application.

    This works by:
    1. Loading the main documentation page
    2. Extracting the window.__NUXT__.state.pages.hierarchy object
    3. Recursively parsing the navigation tree to extract all routes

    Args:
        base_url: Base URL of the documentation (e.g., https://doc.tasq.qarnot.com)

    Returns:
        List of full documentation URLs
    """
    print(f"[route-discovery] Discovering routes from {base_url}...")

    discovered_routes = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # Navigate to the main documentation page
            doc_url = f"{base_url}/documentation/en/home"
            page.goto(doc_url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(3000)  # Wait for Nuxt to hydrate

            # Extract the navigation hierarchy from Nuxt global state
            nuxt_state = page.evaluate("""
                () => {
                    if (window.__NUXT__ && window.__NUXT__.state && window.__NUXT__.state.pages) {
                        return window.__NUXT__.state.pages.hierarchy;
                    }
                    return null;
                }
            """)

            if not nuxt_state:
                print("[route-discovery] ⚠️  Could not find window.__NUXT__.state.pages.hierarchy")
                return []

            print(f"[route-discovery] ✅ Found navigation hierarchy with {len(nuxt_state)} top-level items")

            # Recursively extract all routes
            def extract_routes(items: List[Dict[str, Any]]) -> List[str]:
                """Recursively extract all 'route' fields from the hierarchy."""
                routes = []
                for item in items:
                    if isinstance(item, dict):
                        # Get the route if it exists
                        if 'route' in item and item['route']:
                            routes.append(item['route'])

                        # Recursively process children
                        if 'children' in item and item['children']:
                            routes.extend(extract_routes(item['children']))

                return routes

            relative_routes = extract_routes(nuxt_state)

            # Convert relative routes to full URLs
            doc_base = f"{base_url}/documentation/en"
            discovered_routes = [f"{doc_base}{route}" for route in relative_routes]

            print(f"[route-discovery] ✅ Discovered {len(discovered_routes)} routes")

        except Exception as e:
            print(f"[route-discovery] ❌ Error: {e}")

        finally:
            browser.close()

    return discovered_routes


def save_routes(routes: List[str], output_file: str = "discovered_routes.json") -> None:
    """
    Save discovered routes to a JSON file.

    Args:
        routes: List of discovered URLs
        output_file: Path to output JSON file (relative to project root)
    """
    # Ensure we save to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, output_file)

    data = {
        "total": len(routes),
        "routes": sorted(routes),
        "categories": {
            "getting-started": len([r for r in routes if '/getting-started/' in r]),
            "choosing-your-software": len([r for r in routes if '/choosing-your-software/' in r]),
            "core-concepts": len([r for r in routes if '/core-concepts/' in r]),
            "how-to": len([r for r in routes if '/how-to/' in r]),
            "monitoring-debugging": len([r for r in routes if '/monitoring-debugging/' in r]),
            "other": len([r for r in routes if not any(cat in r for cat in [
                '/getting-started/', '/choosing-your-software/', '/core-concepts/',
                '/how-to/', '/monitoring-debugging/'
            ])])
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[route-discovery] 💾 Saved {len(routes)} routes to {output_path}")
    return output_path


def print_route_summary(routes: List[str]) -> None:
    """Print a summary of discovered routes by category."""
    categories = {
        "Getting Started": [r for r in routes if '/getting-started/' in r],
        "Choosing Software": [r for r in routes if '/choosing-your-software/' in r],
        "Core Concepts": [r for r in routes if '/core-concepts/' in r],
        "How-To Guides": [r for r in routes if '/how-to/' in r],
        "Monitoring & Debugging": [r for r in routes if '/monitoring-debugging/' in r],
    }

    print("\n[route-discovery] 📊 Route Summary:")
    for category, cat_routes in categories.items():
        if cat_routes:
            print(f"  {category}: {len(cat_routes)} routes")

    # Show Docker-specific routes
    docker_routes = [r for r in routes if '/docker' in r.lower()]
    if docker_routes:
        print(f"\n[route-discovery] 🐳 Docker-related routes ({len(docker_routes)}):")
        for route in sorted(docker_routes):
            path = route.replace('https://doc.tasq.qarnot.com/documentation/en', '')
            print(f"    {path}")


def load_discovered_routes(file_path: str = "discovered_routes.json") -> List[str]:
    """
    Load previously discovered routes from JSON file.

    Args:
        file_path: Path to the JSON file (relative to project root)

    Returns:
        List of URLs
    """
    # Look for file in project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    full_path = os.path.join(project_root, file_path)

    if not os.path.exists(full_path):
        return []

    with open(full_path, 'r') as f:
        data = json.load(f)

    return data.get('routes', [])


if __name__ == "__main__":
    # Discover routes
    base_url = "https://doc.tasq.qarnot.com"
    routes = discover_routes_from_nuxt(base_url)

    if routes:
        # Save to file
        output_path = save_routes(routes)

        # Print summary
        print_route_summary(routes)

        print(f"\n[route-discovery] ✅ Done! Routes saved to {output_path}")
        print(f"[route-discovery] 💡 Use these routes in index_builder.py for complete documentation coverage")
    else:
        print("[route-discovery] ❌ No routes discovered")
