"""
Single source of truth for product category definitions and mappings.

All category normalization across the codebase delegates here.
This prevents drift bugs from having the same categories in 3+ places.
"""

import pandas as pd


# =============================================================================
# CANONICAL CATEGORY LIST
# =============================================================================

# Category name → description (used in LLM prompts for filter extraction)
CATEGORY_DESCRIPTIONS = {
    'dock': 'USB-C docks, Thunderbolt docks, laptop docking stations. Use when user needs charging + multiple monitors + ethernet',
    'kvm_switch': 'KVM switches to share monitors/keyboards between computers',
    'kvm_extender': 'KVM extenders (extend KVM over Cat5/fiber)',
    'adapter': 'Video adapters, USB-C to HDMI/VGA/DP adapters',
    'multiport_adapter': 'USB-C multiport adapters (simpler than docks: typically video + a few USB ports, may lack charging or ethernet)',
    'hub': 'USB hubs, port expanders',
    'cable': 'All cables (HDMI, DisplayPort, USB, ethernet, audio)',
    'cable_organizer': 'Cable clips, cable ties, cable management, cord organizers, cable fasteners',
    'fiber_cable': 'Fiber optic cables',
    'ethernet_switch': 'Network switches, gigabit switches',
    'network': 'Network adapters, NICs, SFP modules, media converters',
    'rack': 'Server racks, rack accessories, wall-mount racks, PDUs',
    'storage_enclosure': 'Hard drive enclosures, RAID enclosures, SSD enclosures',
    'display_mount': 'Monitor arm mounts, desk mounts, wall mounts for displays',
    'mount': 'Generic mounts, tablet stands, workstation accessories',
    'computer_card': 'PCIe cards, expansion cards',
    'video_splitter': 'HDMI splitters, DisplayPort splitters. Use when user wants to duplicate/mirror/clone a display to multiple screens, or show the same thing on two or more monitors',
    'video_switch': 'HDMI/video switches (not KVM)',
    'privacy_screen': 'Monitor privacy filters',
    'laptop_lock': 'Laptop cable locks, security locks',
    'power': 'Power adapters, chargers',
    'enclosure': 'Drive enclosures',
    'switch': 'DO NOT USE — use ethernet_switch for network switches, video_switch for HDMI/video switches, or kvm_switch for KVM switches',
    'other': 'Other products',
}

NORMALIZED_CATEGORIES = set(CATEGORY_DESCRIPTIONS.keys())


# =============================================================================
# LLM OUTPUT → NORMALIZED CATEGORY
# =============================================================================

# Maps variations that the LLM might output back to canonical names.
# Used by LLMFilterExtractor._normalize_category()
LLM_CATEGORY_VARIATIONS = {
    # Dock variations
    'docks': 'dock',
    'docking station': 'dock',
    'docking stations': 'dock',
    'usb-c dock': 'dock',
    'thunderbolt dock': 'dock',
    # KVM variations
    'kvm': 'kvm_switch',
    'kvm switch': 'kvm_switch',
    'kvm switches': 'kvm_switch',
    # Adapter variations
    'adapters': 'adapter',
    'video adapter': 'adapter',
    'display adapter': 'adapter',
    # Multiport variations
    'multiport': 'multiport_adapter',
    'multiport adapters': 'multiport_adapter',
    # Hub variations
    'hubs': 'hub',
    'usb hub': 'hub',
    # Cable variations
    'cables': 'cable',
    'ethernet cable': 'cable',
    'network cable': 'cable',
    'hdmi cable': 'cable',
    'displayport cable': 'cable',
    'usb cable': 'cable',
    'usb-c cable': 'cable',
    'thunderbolt cable': 'cable',
    # Media converter / SFP variations (must come before generic fiber)
    'media converter': 'network',
    'fiber media converter': 'network',
    'fiber converter': 'network',
    'sfp': 'network',
    'sfp module': 'network',
    'sfp+': 'network',
    'fiber optic media converter': 'network',
    'fiber optic converter': 'network',
    # Fiber variations
    'fiber': 'fiber_cable',
    'fiber cable': 'fiber_cable',
    'fiber optic': 'fiber_cable',
    # Network switch variations
    'network switch': 'ethernet_switch',
    'gigabit switch': 'ethernet_switch',
    # Rack variations
    'racks': 'rack',
    'server rack': 'rack',
    'racks and enclosures': 'rack',
    # Storage variations
    'storage': 'storage_enclosure',
    'enclosures': 'storage_enclosure',
    'drive enclosure': 'storage_enclosure',
    'data storage': 'storage_enclosure',
    # Display mount variations
    'display mount': 'display_mount',
    'display mounts': 'display_mount',
    'monitor mount': 'display_mount',
    'monitor arm': 'display_mount',
    'tv mount': 'display_mount',
    'desk mount': 'display_mount',
    # Generic mount variations
    'mounts': 'mount',
    'tablet stand': 'mount',
    # KVM extender variations
    'kvm extender': 'kvm_extender',
    'kvm extenders': 'kvm_extender',
    # Computer card variations
    'computer cards': 'computer_card',
    'pcie card': 'computer_card',
    'expansion card': 'computer_card',
    # Video splitter variations
    'splitter': 'video_splitter',
    'hdmi splitter': 'video_splitter',
    # Video switch variations (not KVM)
    'hdmi switch': 'video_switch',
    # Privacy screen variations
    'privacy filter': 'privacy_screen',
    # Cable organizer variations
    'cable organizer': 'cable_organizer',
    'cable organizers': 'cable_organizer',
    'cable clip': 'cable_organizer',
    'cable clips': 'cable_organizer',
    'cable tie': 'cable_organizer',
    'cable ties': 'cable_organizer',
    'cable management': 'cable_organizer',
    'cord organizer': 'cable_organizer',
    'cable fastener': 'cable_organizer',
    # Laptop lock variations
    'laptop lock': 'laptop_lock',
    'laptop locks': 'laptop_lock',
    'cable lock': 'laptop_lock',
    'security lock': 'laptop_lock',
}


# =============================================================================
# SEARCH HISTORY KEYWORDS
# =============================================================================

# Short keyword lists for matching user queries to categories.
# Used by orchestrator._extract_category_hint()
CATEGORY_KEYWORDS = {
    'dock': ['dock', 'docking'],
    'hub': ['hub', 'usb hub'],
    'cable': ['cable', 'cables'],
    'fiber_cable': ['fiber', 'fiber optic'],
    'adapter': ['adapter', 'adapters'],
    'multiport_adapter': ['multiport'],
    'kvm_switch': ['kvm'],
    'kvm_extender': ['kvm extender'],
    'ethernet_switch': ['ethernet switch', 'network switch'],
    'network': ['media converter', 'sfp', 'network card', 'nic'],
    'switch': ['switch', 'switches'],
    'rack': ['rack', 'server rack'],
    'storage_enclosure': ['enclosure', 'drive bay', 'drive enclosure'],
    'display_mount': ['monitor mount', 'tv mount', 'monitor arm'],
    'mount': ['mount', 'mounting'],
    'computer_card': ['pcie', 'expansion card'],
    'video_splitter': ['splitter'],
    'video_switch': ['video switch', 'hdmi switch'],
    'privacy_screen': ['privacy screen', 'privacy filter'],
    'cable_organizer': ['cable clip', 'cable tie', 'cable organizer', 'cable management', 'cord organizer'],
    'laptop_lock': ['laptop lock', 'cable lock', 'security lock'],
    'power': ['pdu', 'ups', 'power'],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_llm_category(category: str) -> str:
    """
    Normalize an LLM-output category name to its canonical form.

    Args:
        category: Category string from LLM output (e.g. 'docks', 'docking station')

    Returns:
        Canonical category name (e.g. 'dock')
    """
    category_lower = category.lower().strip()

    # Already canonical
    if category_lower in NORMALIZED_CATEGORIES:
        return category_lower

    # Check variations
    return LLM_CATEGORY_VARIATIONS.get(category_lower, category_lower)


def determine_category(row: pd.Series) -> str:
    """
    Determine normalized product category from ItemCategory and ItemSubCategory columns.
    Maps to standard categories: cable, adapter, dock, hub, switch, etc.

    This is the single source of truth for Excel data → category mapping.
    """
    category = row.get('ItemCategory')
    sub_category = row.get('ItemSubCategory')

    cat_str = ''
    if pd.notna(category):
        cat_str += str(category).lower()
    if pd.notna(sub_category):
        cat_str += ' ' + str(sub_category).lower()

    # Check connectors for special case handling
    interface_a = str(row.get('Connector_A', '')).lower()
    interface_b = str(row.get('Connector_B', '')).lower()

    # USB-C to video output are cables, not adapters
    if ('usb' in interface_a and 'type-c' in interface_a) or ('usb-c' in interface_a):
        if any(video in interface_b for video in ['hdmi', 'displayport', 'display port']):
            return 'cable'

    # Map to standard categories
    # Order matters: more specific categories first

    # Fiber cables (before generic 'cable' check)
    if 'fiber' in cat_str:
        return 'fiber_cable'

    # Storage/drive enclosures (before generic 'enclosure' check)
    if 'external drive' in cat_str or 'drive enclosure' in cat_str:
        return 'storage_enclosure'
    if 'data storage' in cat_str:
        if 'enclosure' in cat_str:
            return 'storage_enclosure'
        if 'drive' in cat_str and 'adapter' not in cat_str and 'converter' not in cat_str:
            return 'storage_enclosure'

    # Cable organizers (before generic 'cable' check)
    if 'cable organizer' in cat_str or 'cable fastener' in cat_str:
        return 'cable_organizer'

    if 'cable' in cat_str:
        return 'cable'
    # Racks must be checked BEFORE enclosures ("Racks and Enclosures" contains both)
    elif 'rack' in cat_str:
        return 'rack'
    # Computer cards must be checked BEFORE adapters ("Computer Cards and Adapters" contains both)
    elif 'computer card' in cat_str or ('card' in cat_str and 'adapter' in cat_str):
        return 'computer_card'
    # Multiport adapters must be checked BEFORE generic adapters
    elif 'multiport' in cat_str:
        return 'multiport_adapter'
    elif 'adapter' in cat_str or 'converter' in cat_str:
        return 'adapter'
    elif 'dock' in cat_str or 'docking' in cat_str:
        return 'dock'
    elif 'hub' in cat_str:
        return 'hub'
    elif 'ethernet switch' in cat_str:
        return 'ethernet_switch'
    # KVM cables/extenders must be checked BEFORE generic 'kvm' pattern
    elif 'kvm cable' in cat_str:
        return 'cable'
    elif 'kvm extender' in cat_str:
        return 'kvm_extender'
    elif 'kvm' in cat_str:
        return 'kvm_switch'
    elif 'video switch' in cat_str:
        return 'video_switch'
    elif 'switch' in cat_str:
        return 'switch'
    elif 'enclosure' in cat_str:
        return 'enclosure'
    # Display mounts (monitor/TV) must be checked BEFORE generic 'mount'
    elif 'display mount' in cat_str or 'monitor mount' in cat_str or 'tv mount' in cat_str:
        return 'display_mount'
    elif 'monitor' in cat_str and 'mount' in cat_str:
        return 'display_mount'
    elif 'mount' in cat_str:
        return 'mount'
    elif 'privacy' in cat_str or 'screen filter' in cat_str:
        return 'privacy_screen'
    elif 'video splitter' in cat_str or ('splitter' in cat_str and 'video' in cat_str):
        return 'video_splitter'
    elif 'lock' in cat_str:
        return 'laptop_lock'
    elif 'power' in cat_str:
        return 'power'
    elif 'network' in cat_str:
        return 'network'
    else:
        return 'other'


def get_category_keywords(category: str) -> list:
    """Get search keywords for a category."""
    return CATEGORY_KEYWORDS.get(category, [category.lower()])
