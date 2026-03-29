"""
Search engine configuration constants.

These constants were originally class attributes on ProductSearchEngine.
They are defined at module level so they can be imported independently
by any module that needs them.

Constants:
    GENERIC_CATEGORIES — Category names that require exact match (no suffix matching).
        Prevents generic names like "cable" from incorrectly matching more specific
        subcategories (e.g., "fiber cable").

    CONNECTOR_MATCH_CONFIG — Data-driven connector matching configuration.
        Each entry defines how to match a search term against product connector data:
        search_aliases, product_patterns (regex), and exclusions.

    FILTER_DISPATCH — Dispatch table mapping filter_dict keys to
        (method_name, weight, is_gate). Gates are mandatory pass/fail filters;
        scored filters contribute to product relevance score.

    AUXILIARY_KEYS — Keys consumed by parent matchers that are not scored independently.
"""

# Categories that require exact match (no suffix matching).
# These are "generic" category names that could incorrectly match
# more specific categories (e.g., "cable" matching "fiber cable").
GENERIC_CATEGORIES = {'cable', 'adapter', 'switch', 'card', 'enclosure', 'network', 'mount'}

# Data-driven connector matching config.
# Each entry defines how to match a search term against product connector data:
#   search_aliases  — terms the user/LLM might use for this connector type
#   product_patterns — regex patterns to find this connector in product data
#   exclusions — substrings in product data that mean it's NOT this connector
# To add a new connector type, just add a new entry here.
CONNECTOR_MATCH_CONFIG = {
    'usb-c': {
        'search_aliases': ['usb-c', 'usb c', 'type-c', 'type c', 'usbc'],
        'product_patterns': [r'usb[\s-]?c', r'usb\s+type[\s-]?c', r'type[\s-]?c'],
        'exclusions': [],
    },
    'usb-b': {
        'search_aliases': ['usb-b', 'usb b', 'type-b', 'type b', 'usbb'],
        'product_patterns': [r'usb[\s-]?b', r'usb\s+type[\s-]?b', r'type[\s-]?b'],
        'exclusions': [],
    },
    'displayport': {
        'search_aliases': ['displayport', 'display port', 'dp'],
        'product_patterns': [r'displayport'],
        'exclusions': ['alt mode', 'alternate mode', 'mini'],
    },
    'mini-displayport': {
        'search_aliases': ['mini-displayport', 'mini displayport', 'mdp'],
        'product_patterns': [r'mini[\s-]?displayport'],
        'exclusions': [],
    },
    'hdmi': {
        'search_aliases': ['hdmi'],
        'product_patterns': [r'hdmi'],
        'exclusions': ['mini', 'micro'],
    },
    'mini-hdmi': {
        'search_aliases': ['mini-hdmi', 'mini hdmi'],
        'product_patterns': [r'mini[\s-]?hdmi'],
        'exclusions': [],
    },
    'micro-hdmi': {
        'search_aliases': ['micro-hdmi', 'micro hdmi'],
        'product_patterns': [r'micro[\s-]?hdmi'],
        'exclusions': [],
    },
    'ethernet': {
        'search_aliases': ['rj45', 'rj-45', 'ethernet'],
        'product_patterns': [r'rj[\s-]?45', r'ethernet'],
        'exclusions': [],
    },
    'vga': {
        'search_aliases': ['vga'],
        'product_patterns': [r'vga', r'd-sub'],
        'exclusions': [],
    },
    'dvi': {
        'search_aliases': ['dvi'],
        'product_patterns': [r'dvi'],
        'exclusions': [],
    },
    'thunderbolt': {
        'search_aliases': ['thunderbolt'],
        'product_patterns': [r'thunderbolt'],
        'exclusions': [],
    },
    'nema': {
        'search_aliases': ['nema', 'nema 5-15', 'ac power', 'ac', 'power plug'],
        'product_patterns': [r'nema', r'power.*north america'],
        'exclusions': ['power delivery', 'usb'],
    },
    'iec': {
        'search_aliases': ['iec', 'iec 60320', 'c13', 'c14', 'c19', 'c20'],
        'product_patterns': [r'iec\s*60320', r'c1[34]\s*power', r'c[12]0\s*power'],
        'exclusions': [],
    },
}

# Dispatch table mapping filter_dict keys to (method_name, weight, is_gate).
# Gates: mandatory pass/fail filters (score 0 if fail).
# Scored: weighted filters that contribute to product relevance score.
FILTER_DISPATCH = {
    # --- Mandatory gates (pass/fail) ---
    'category':                ('_matches_category',              0,  True),
    'connector_from':          ('_matches_connectors',            0,  True),
    'connector_to':            ('_matches_connectors',            0,  True),
    # --- Product-defining specs (gates: mismatch = wrong product) ---
    'cable_type':              ('_matches_cable_type',            15, True),
    'kvm_video_type':          ('_matches_kvm_video_type',        12, True),
    'keywords':                ('_matches_keywords',              10, True),
    'port_count':              ('_matches_port_count',            10, True),
    'min_monitors':            ('_matches_min_monitors',          10, True),
    'screen_size':             ('_matches_screen_size',           10, True),
    'bay_count':               ('_matches_bay_count',             10, True),
    'rack_height':             ('_matches_rack_height',           10, True),
    'drive_size':              ('_matches_drive_size',            10, True),
    'usb_version':             ('_matches_usb_version',           10, False),
    'thunderbolt_version':     ('_matches_thunderbolt_version',   10, True),
    'requested_network_speed': ('_matches_network_speed',         10, False),
    # --- Medium weight (important preferences) ---
    'features':                ('_matches_features',              10, False),
    'required_port_types':     ('_matches_required_port_types',    8, False),
    'requested_refresh_rate':  ('_matches_refresh_rate',           8, False),
    'length':                  ('_matches_length',                 6, False),
    'requested_power_wattage': ('_matches_power_wattage',          6, True),
    # --- Low weight (soft preferences) ---
    'color':                   ('_matches_color',                  4, False),
}

# Keys consumed by parent matchers (not scored independently).
AUXILIARY_KEYS = {'same_connector', 'length_unit', 'length_preference'}
