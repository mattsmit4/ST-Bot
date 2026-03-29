"""
LLM-Based Filter Extraction - Semantic Query Understanding

Uses OpenAI to extract structured filters from natural language queries
when rule-based extraction fails or returns ambiguous results.

This module provides:
- Semantic understanding of complex queries
- Disambiguation of context-dependent values (e.g., "32 inch" vs "32ft")
- Category inference from natural language descriptions
- Fallback capability when rule-based extraction returns 0 results

Design:
- Hybrid approach: LLM only called when needed (saves cost/latency)
- Structured output: Returns same filter format as rule-based extractor
- Graceful degradation: Falls back to rule-based if LLM fails
"""

import os
import json
import re
import logging
from typing import Optional, Dict, Any, List
from core.models import SearchFilters, LengthPreference, LLMExtractionResult
from core.api_retry import RetryHandler, RetryConfig
from core.openai_client import get_openai_client

# Module-level logger
_logger = logging.getLogger(__name__)


# Generate category list from single source of truth
from core.category_config import CATEGORY_DESCRIPTIONS
_CATEGORY_LIST = "\n".join(
    f'- "{name}" - {desc}'
    for name, desc in CATEGORY_DESCRIPTIONS.items()
    if name != 'other'
)

# System prompt for filter extraction
# Category names generated from core.category_config (single source of truth)
_EXTRACTION_PROMPT_TEMPLATE = """You are a product search filter extractor for StarTech.com, a computer hardware retailer.

Given a user's natural language query, extract structured search filters.

## IMPORTANT: Product Categories
**CRITICAL**: If the user's query explicitly mentions a product type word, you MUST set product_category. Match the word to the closest category below (e.g., "cables" → "cable", "docks" → "dock", "splitter" → "video_splitter", "hub" → "hub", "mounts" → "display_mount", etc.).

You MUST use one of these EXACT normalized category names:
{category_list}

## Extraction Rules

1. **Connectors** - What connectors are involved?
   - connector_from: Source/host side (e.g., USB-C, DisplayPort, HDMI)
   - connector_to: Destination side
   - For docks and hubs: only set connector_from (the host connection)
   - "USB-C hub" → connector_from: "USB-C", product_category: "hub"
   - "USB-A hub" → connector_from: "USB-A", product_category: "hub"
   - IMPORTANT: Preserve size qualifiers — "Mini DisplayPort", "Mini HDMI", "Micro HDMI"
     are DISTINCT connector types from "DisplayPort" and "HDMI"
   - "Mini DisplayPort cable" → connector_from: "Mini DisplayPort", connector_to: "Mini DisplayPort"
   - "Micro HDMI to HDMI" → connector_from: "Micro HDMI", connector_to: "HDMI"
   - **Same-connector cables**: When the query says "[connector] cables" without specifying a
     different destination (e.g., "USB-C cables", "Thunderbolt cables", "HDMI cables"), set
     BOTH connector_from AND connector_to to that connector. This filters out adapter cables.
     - "USB-C cables" → connector_from: "USB-C", connector_to: "USB-C"
     - "Thunderbolt cables" → connector_from: "Thunderbolt", connector_to: "Thunderbolt"
     - "USB-C to HDMI cable" → connector_from: "USB-C", connector_to: "HDMI" (different destination = NOT same)
   - If the user needs MULTIPLE host types (e.g., "works with USB-C and USB-A laptops"),
     put the primary in connector_from and add the secondary as a keyword.
     Example: "dock for USB-C and USB-A" → connector_from: "USB-C", keywords: ["USB-A compatible"]
   - **Power/AC cables**: Use connector_from/to with "NEMA" or "IEC" — these are the actual
     connector types for power cables. Do NOT use "AC" as a connector.
     - "power cable" → connector_from: "NEMA", product_category: "cable", keywords: ["power"]
     - "C13 power cord" → connector_from: "IEC", connector_to: "IEC", keywords: ["C13"]
     - "server power cable" → connector_from: "NEMA", connector_to: "IEC", keywords: ["power"]

2. **Length** - Cable length in feet
   - IMPORTANT: "32 inch monitor" is screen size, NOT cable length!
   - Only extract length for actual cable/product dimensions
   - Extract length_preference when user specifies:
     - "at least 15 feet" → length_ft: 15, length_preference: "exact_or_longer"
     - "no longer than 6ft" → length_ft: 6, length_preference: "exact_or_shorter"
     - "15 foot cable" → length_ft: 15, length_preference: null (default)
   - For RANGE queries, use length_min_ft and length_max_ft instead of length_ft:
     - "between 1 and 2 meters" → length_min_ft: 3.28, length_max_ft: 6.56, length_ft: null
     - "1 to 3 feet" → length_min_ft: 1, length_max_ft: 3, length_ft: null
   - For KVM extenders/switches, distance is NOT cable length — put it in keywords instead.
     Example: "extend HDMI 150 feet" → length_ft: null, keywords: ["150ft"]

3. **Features** - Broad technical CAPABILITIES only: 4K, 8K, PoE, Power Delivery, shielded, HDCP, HDR
   - Features match the product's capability tags — keep them categorical
   - Do NOT put specific product descriptors in features — use keywords instead:
     - "industrial-grade" → keywords (product descriptor, not a capability)
     - "DIN rail mountable" → keywords (form factor, matches product names)
     - Temperature ranges like "-40 to +75C" → keywords (spec, not a capability)
     - Wattage like "90W" → requested_power_wattage (not features, not keywords)

4. **Keywords** - ONLY technical product terms that appear in metadata
   - GOOD keywords: "Cat6a", "gigabit", "RAID", "PoE", "slim"
   - BAD keywords: "dual monitor" (use min_monitors instead), "extra USB ports" (natural language)
   - Keywords are for matching product SKUs, names, and specs - NOT for describing user needs
   - For docks and KVMs: usually leave keywords EMPTY and use min_monitors/port_count instead
   - NEVER put a value in keywords when a dedicated field exists. Use the field instead:
     - Thunderbolt 3/4 → thunderbolt_version (not keywords)
     - 60Hz, 144Hz → requested_refresh_rate (not keywords)
     - 90W, 100W → requested_power_wattage (not keywords, and NOT features either)
     - gigabit, 10Gbps → network_speed_mbps (not keywords) — for cables
     - dual/triple monitor → min_monitors (not keywords)
   - Keywords should only contain terms with NO dedicated field (e.g., "single mode", "Kensington lock", "audio jack")
   - **For the "network" category**: ALWAYS extract the product subtype as a keyword to distinguish between different network product types:
     - "PoE injector" → keywords: ["injector"]
     - "PoE extender" → keywords: ["extender"]
     - "media converter" → keywords: ["media converter"]
     - "network card" or "NIC" → keywords: ["network card"]
     - "SFP module" → keywords: ["SFP"]
     - IEEE standards (802.3bt, 802.3at, 802.3af) → keywords (no dedicated field for these)

5. **port_count** - Number of ports for hubs/switches/KVMs/splitters
   - "3 computers to 2 monitors" for KVM = port_count: 3 (computer ports)
   - "2-port KVM switch" → port_count: 2 (NOT min_monitors — "port" means computer ports)
   - For video splitters: "N outputs" or "1-to-N" = port_count (output ports)
     - "2 outputs" → port_count: 2. "1 to 4 splitter" → port_count: 4

6. **min_monitors** - Number of monitors to support (for docks, KVMs, display mounts ONLY)
   - Use this instead of keywords like "dual monitor"
   - "dual monitor mount" → min_monitors: 2
   - Do NOT set min_monitors from "X-port" — that refers to port_count
   - Do NOT use for video_splitter or video_switch — use port_count instead

7. **network_speed_mbps** - Requested network speed for ethernet/network cables
   - Extract speed in Mbps: "10Gbps" = 10000, "1Gbps"/"gigabit" = 1000, "100Mbps" = 100
   - IMPORTANT: When user asks for "10Gbps" or "10 gigabit", this requires Cat6a or Cat7 cables
   - Only applies to network/ethernet cables, not other cable types

8. **thunderbolt_version** - For Thunderbolt docks/cables (3 or 4)
   - "Thunderbolt 4 dock" → thunderbolt_version: 4
   - "TB3 cable" → thunderbolt_version: 3
   - IMPORTANT: Do NOT confuse Thunderbolt version with connector type
   - Thunderbolt docks use USB-C connector but require specific Thunderbolt support

9. **kvm_video_type** - Video interface for KVM switches and KVM extenders ONLY
   - "HDMI KVM switch" → kvm_video_type: "HDMI"
   - "extend HDMI signal" → kvm_video_type: "HDMI", product_category: "kvm_extender"
   - "DisplayPort KVM" → kvm_video_type: "DisplayPort"
   - Use this INSTEAD of connector_from/connector_to for KVM products
   - Do NOT use for video_splitter or video_switch — use connector_from instead
     - "HDMI splitter" → connector_from: "HDMI", kvm_video_type: null

10. **Fiber optic** - IMPORTANT distinction:
   - "fiber optic cable", "fiber patch cable", "fiber jumper" → category: "fiber_cable"
   - "fiber optic media converter", "SFP module", "fiber transceiver" → category: "network"
   - Media converters and SFP modules are NETWORK devices, not cables
   - For fiber cables: extract mode and classification as keywords:
     - "single mode" / "singlemode" / "SM" → keywords: ["single mode"]
     - "multi mode" / "multimode" / "MM" → keywords: ["multimode"]
     - "OM3", "OM4", "OS2" → add to keywords (e.g., keywords: ["single mode", "OS2"])

11. **rack_height** - For rack/enclosure queries, extract the U-height
   - "6U rack" → rack_height: 6
   - "12U enclosure" → rack_height: 12
   - This represents MINIMUM height — search will return this size and larger

12. **Device-to-connector inference** — When the user mentions devices instead of connectors:
   - Laptops, MacBooks, Chromebooks, iPads → connector_from: "USB-C"
   - TVs, monitors, projectors → connector_to: "HDMI"
   - Printers, scanners → connector_to: "USB-B" (ALWAYS use "USB-B" exactly, not "USB"), product_category: "cable"
   - Desktop computers → connector_from: "USB-C" (modern) or null if unclear
   - "connect X to Y" or "hook up X to Y" → product_category: "cable"
   - ALWAYS infer connectors from devices when no explicit connector is mentioned

13. **"Already in use" / "occupied" ports** — When a connector is mentioned as a constraint
   (already in use, taken, occupied, no more X ports) rather than the desired product connector:
   - Do NOT set connector_from to that connector — it's unavailable, not desired
   - Set product_category to "adapter" or "multiport_adapter"
   - Set connector_to to the destination type (usually "HDMI" for monitor additions)
   - Leave connector_from as null — user has unspecified alternative ports
   - Example: "HDMI port already in use" → category: "adapter", connector_from: null, connector_to: "HDMI"

14. **Video capability (DP Alt Mode)** — For USB-C to USB-C cable queries:
   - If user mentions monitors, displays, screens, projectors, daisy-chain, or video → add "DP Alt Mode" to features
   - "USB-C cable for daisy-chaining monitors" → features: ["DP Alt Mode"]
   - "USB-C video cable" → features: ["DP Alt Mode"]
   - "USB-C charging cable" → do NOT add DP Alt Mode (data/power only)
   - DP 1.4, DP 1.2, DisplayPort Alt Mode → features: ["DP Alt Mode"]

15. **requested_power_wattage** - Power delivery wattage for charging
   - "90W charging" → requested_power_wattage: 90
   - "100W power delivery" → requested_power_wattage: 100
   - Only extract when user specifies a specific wattage

16. **requested_refresh_rate** - Display refresh rate in Hz
   - "60Hz" → requested_refresh_rate: 60
   - "144Hz gaming monitor" → requested_refresh_rate: 144
   - Only extract when user specifies a specific refresh rate

17. **bay_count** - Number of drive bays for storage enclosures
   - "4-bay enclosure" → bay_count: 4
   - "dual drive enclosure" → bay_count: 2
   - "RAID enclosure for 4 drives" → bay_count: 4
   - Only extract for storage enclosure queries

18. **screen_size** - Screen size in inches for privacy screens
   - "15 inch privacy screen" → screen_size: 15.0
   - "privacy filter for 13.3 inch laptop" → screen_size: 13.3
   - "24 inch monitor privacy filter" → screen_size: 24.0
   - Only extract for privacy screen/filter queries

19. **drive_size** - Drive form factor for storage enclosures
   - "2.5 inch drive enclosure" → drive_size: "2.5"
   - "3.5 inch hard drive bay" → drive_size: "3.5"
   - "M.2 NVMe enclosure" → drive_size: "M.2 NVMe"
   - "mSATA enclosure" → drive_size: "mSATA"
   - "both 2.5 and 3.5 inch drives" → drive_size: "2.5, 3.5"
   - Only extract for storage enclosure queries

20. **usb_version** - USB specification version
   - "USB 3.0 hub" → usb_version: "USB 3.0"
   - "USB 3.2 Gen 2 adapter" → usb_version: "USB 3.2 Gen 2"
   - "USB 2.0 hub" → usb_version: "USB 2.0"
   - "USB-C" alone is a connector type, NOT a USB version — do not set usb_version for connector-only mentions

Respond ONLY with valid JSON:
{
    "product_category": "exact category name or null",
    "connector_from": "string or null",
    "connector_to": "string or null",
    "length_ft": "number or null (single length, NOT for ranges)",
    "length_min_ft": "number or null (range query minimum, in feet)",
    "length_max_ft": "number or null (range query maximum, in feet)",
    "length_preference": "'exact_or_longer' | 'exact_or_shorter' | null",
    "color": "string or null",
    "features": ["list"],
    "keywords": ["list"],
    "min_monitors": "number or null",
    "port_count": "number or null",
    "network_speed_mbps": "number or null (10000 for 10Gbps, 1000 for 1Gbps)",
    "thunderbolt_version": "number or null (3 or 4)",
    "kvm_video_type": "string or null (HDMI, DisplayPort, VGA, DVI - for KVM switches/extenders)",
    "rack_height": "number or null (U-height, e.g. 6 for 6U rack)",
    "requested_power_wattage": "number or null (wattage, e.g. 90 for 90W charging)",
    "requested_refresh_rate": "number or null (Hz, e.g. 60 for 60Hz)",
    "bay_count": "number or null (drive bays, e.g. 4 for 4-bay enclosure)",
    "screen_size": "number or null (inches, e.g. 15.6 for 15.6 inch privacy screen)",
    "drive_size": "string or null ('2.5', '3.5', 'M.2 NVMe', 'M.2 SATA', 'mSATA')",
    "usb_version": "string or null ('USB 2.0', 'USB 3.0', 'USB 3.2 Gen 2')",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}

## Examples

Query: "I need a dock for my MacBook with 2 monitors and charging"
{
    "product_category": "dock",
    "connector_from": "USB-C",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": ["Power Delivery", "4K"],
    "keywords": [],
    "min_monitors": 2,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": null,
    "confidence": 0.95,
    "reasoning": "USB-C dock for MacBook with dual monitor support and charging via Power Delivery."
}

Query: "Thunderbolt 4 dock with triple monitor 4K 60Hz and 100W charging"
{
    "product_category": "dock",
    "connector_from": "Thunderbolt",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": ["Power Delivery", "4K"],
    "keywords": [],
    "min_monitors": 3,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": 4,
    "requested_power_wattage": 100,
    "requested_refresh_rate": 60,
    "confidence": 0.95,
    "reasoning": "Thunderbolt 4 dock with triple monitor support at 4K 60Hz. 100W charging requires Power Delivery."
}

Query: "switch between 3 computers and 2 monitors"
{
    "product_category": "kvm_switch",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": 2,
    "port_count": 3,
    "confidence": 0.95,
    "reasoning": "KVM switch to connect 3 computers to 2 monitors."
}

Query: "2-port KVM switch with 4K 60Hz DisplayPort"
{
    "product_category": "kvm_switch",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": ["4K"],
    "requested_refresh_rate": 60,
    "keywords": [],
    "min_monitors": null,
    "port_count": 2,
    "kvm_video_type": "DisplayPort",
    "confidence": 0.95,
    "reasoning": "2-port means 2 computer ports, NOT 2 monitors. DisplayPort video interface."
}

Query: "USB-C to VGA for projector 30 feet away"
{
    "product_category": "adapter",
    "connector_from": "USB-C",
    "connector_to": "VGA",
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "confidence": 0.9,
    "reasoning": "USB-C to VGA adapter. 30 feet is the distance - may need a separate long VGA cable."
}

Query: "200ft network cable for gigabit"
{
    "product_category": "cable",
    "connector_from": "RJ45",
    "connector_to": "RJ45",
    "length_ft": 200,
    "color": null,
    "features": [],
    "keywords": ["ethernet"],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": 1000,
    "confidence": 0.9,
    "reasoning": "Long ethernet cable for gigabit (1Gbps) network."
}

Query: "10Gbps ethernet cables for office 250ft"
{
    "product_category": "cable",
    "connector_from": "RJ45",
    "connector_to": "RJ45",
    "length_ft": 250,
    "color": null,
    "features": ["shielded"],
    "keywords": ["ethernet"],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": 10000,
    "confidence": 0.95,
    "reasoning": "10Gbps ethernet requires Cat6a or Cat7 cables. Shielded recommended for office environments."
}

Query: "server rack accessories"
{
    "product_category": "rack",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": ["accessories", "server"],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": null,
    "rack_height": null,
    "confidence": 0.85,
    "reasoning": "Server rack accessories and components."
}

Query: "6U wall-mount rack enclosure"
{
    "product_category": "rack",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": ["wall mount"],
    "keywords": ["enclosure"],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": null,
    "rack_height": 6,
    "confidence": 0.95,
    "reasoning": "6U wall-mount rack enclosure. rack_height=6 means show 6U or larger."
}

Query: "15 meter single mode fiber optic cable with LC connectors"
{
    "product_category": "fiber_cable",
    "connector_from": "LC",
    "connector_to": "LC",
    "length_ft": 49,
    "color": null,
    "features": [],
    "keywords": ["single mode"],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": null,
    "rack_height": null,
    "confidence": 0.95,
    "reasoning": "Fiber cable with LC connectors on both ends. 15m ≈ 49ft. 'Single mode' extracted as keyword to filter fiber type."
}

Query: "fiber optic media converter gigabit"
{
    "product_category": "network",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": ["media converter", "fiber", "gigabit"],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": 1000,
    "thunderbolt_version": null,
    "rack_height": null,
    "confidence": 0.95,
    "reasoning": "Fiber optic media converter is a NETWORK device, not a cable."
}

Query: "gigabit PoE injector rated for 60W"
{
    "product_category": "network",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": ["PoE"],
    "keywords": ["injector"],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": 1000,
    "requested_power_wattage": 60,
    "confidence": 0.9,
    "reasoning": "PoE injector is a network device. 'injector' as keyword to distinguish from network cards/converters. 60W in requested_power_wattage NOT features."
}

Query: "RAID enclosure for 4 drives"
{
    "product_category": "storage_enclosure",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": ["RAID"],
    "keywords": ["enclosure"],
    "min_monitors": null,
    "port_count": null,
    "bay_count": 4,
    "drive_size": null,
    "confidence": 0.9,
    "reasoning": "Storage enclosure with RAID support for 4 drives."
}

Query: "2.5 inch USB 3.0 drive enclosure"
{
    "product_category": "storage_enclosure",
    "connector_from": "USB-A",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": ["enclosure"],
    "min_monitors": null,
    "port_count": null,
    "bay_count": null,
    "drive_size": "2.5",
    "usb_version": "USB 3.0",
    "confidence": 0.95,
    "reasoning": "2.5 inch drive enclosure with USB 3.0 interface."
}

Query: "4K HDMI splitter with 2 outputs"
{
    "product_category": "video_splitter",
    "connector_from": "HDMI",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": ["4K"],
    "keywords": [],
    "min_monitors": null,
    "port_count": 2,
    "kvm_video_type": null,
    "confidence": 0.95,
    "reasoning": "HDMI video splitter with 2 output ports. 4K as feature. NOT min_monitors (splitters use port_count). NOT kvm_video_type (splitters use connector_from)."
}

Query: "privacy screen for 15.6 inch laptop"
{
    "product_category": "privacy_screen",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": ["laptop"],
    "min_monitors": null,
    "port_count": null,
    "screen_size": 15.6,
    "confidence": 0.95,
    "reasoning": "Privacy screen for 15.6 inch laptop display."
}

Query: "cable clips for organizing desk cables"
{
    "product_category": "cable_organizer",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": ["clip"],
    "min_monitors": null,
    "port_count": null,
    "confidence": 0.9,
    "reasoning": "User wants cable organizer clips for cable management."
}

Query: "USB 3.2 Gen 2 hub with 7 ports"
{
    "product_category": "hub",
    "connector_from": null,
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": 7,
    "usb_version": "USB 3.2 Gen 2",
    "confidence": 0.95,
    "reasoning": "7-port USB hub with USB 3.2 Gen 2 (10Gbps) speed."
}

Query: "USB-C hub with at least 4 USB-A ports and an SD card reader"
{
    "product_category": "hub",
    "connector_from": "USB-C",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": ["SD card reader"],
    "min_monitors": null,
    "port_count": 4,
    "usb_version": null,
    "confidence": 0.90,
    "reasoning": "USB-C hub (host connection) with minimum 4 USB-A ports and SD card reader."
}

## Device-Based Queries (infer connectors from devices)

Query: "What cable do I need to connect my laptop to my TV?"
{
    "product_category": "cable",
    "connector_from": "USB-C",
    "connector_to": "HDMI",
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": null,
    "rack_height": null,
    "confidence": 0.85,
    "reasoning": "Laptop typically has USB-C, TV typically has HDMI. Cable category for connect X to Y."
}

Query: "I need to hook up my MacBook to a projector"
{
    "product_category": "cable",
    "connector_from": "USB-C",
    "connector_to": "HDMI",
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": null,
    "rack_height": null,
    "confidence": 0.85,
    "reasoning": "MacBook uses USB-C/Thunderbolt. Projectors commonly use HDMI or VGA. Default to HDMI."
}

## Simple Product Type Queries (product type word = MUST set category)

Query: "Do you sell Thunderbolt cables?"
{
    "product_category": "cable",
    "connector_from": "Thunderbolt",
    "connector_to": "Thunderbolt",
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": null,
    "confidence": 0.90,
    "reasoning": "User explicitly says 'cables' with one connector type — set both connector_from and connector_to to Thunderbolt (same-connector cable)."
}

Query: "USB-C cables"
{
    "product_category": "cable",
    "connector_from": "USB-C",
    "connector_to": "USB-C",
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "confidence": 0.90,
    "reasoning": "User says 'cables' with one connector type — set both connector_from and connector_to to USB-C (same-connector cable). NOT USB-C-to-HDMI adapters."
}

Query: "USB-C adapters"
{
    "product_category": "adapter",
    "connector_from": "USB-C",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "confidence": 0.90,
    "reasoning": "User explicitly says 'adapters' — set product_category. USB-C is the host connector. No connector_to since destination unknown."
}

## Short/Conversational Queries
Sometimes users mention just a connector type without full product context.
IMPORTANT: If a connector is MENTIONED, extract it even if the query is short.
The connector itself is valuable filter context for searching.

Query: "Or maybe Thunderbolt?"
{
    "product_category": null,
    "connector_from": "Thunderbolt",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "confidence": 0.7,
    "reasoning": "User mentioned Thunderbolt - extract as connector to search for Thunderbolt products."
}

Query: "USB-C I think"
{
    "product_category": null,
    "connector_from": "USB-C",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "confidence": 0.7,
    "reasoning": "User mentioned USB-C - extract as connector to find USB-C products."
}

Query: "What about HDMI?"
{
    "product_category": null,
    "connector_from": "HDMI",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "confidence": 0.7,
    "reasoning": "User asking about HDMI - extract connector to search HDMI products."
}

Query: "DisplayPort?"
{
    "product_category": null,
    "connector_from": "DisplayPort",
    "connector_to": null,
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": null,
    "port_count": null,
    "confidence": 0.7,
    "reasoning": "Single connector mention - extract to search DisplayPort products."
}

## "Port Already In Use" (connector is a constraint, not desired)

Query: "I need to add a second monitor to my laptop but it only has one HDMI port and that's already in use. What are my options?"
{
    "product_category": "adapter",
    "connector_from": null,
    "connector_to": "HDMI",
    "length_ft": null,
    "color": null,
    "features": [],
    "keywords": [],
    "min_monitors": 2,
    "port_count": null,
    "network_speed_mbps": null,
    "thunderbolt_version": null,
    "rack_height": null,
    "confidence": 0.9,
    "reasoning": "HDMI is mentioned as already in use (constraint, not desired connector). User needs adapter to add second display. connector_from: null (unknown available port), connector_to: HDMI (monitor output), category: adapter, min_monitors: 2."
}
"""

# Inject category list into prompt template
EXTRACTION_PROMPT = _EXTRACTION_PROMPT_TEMPLATE.replace('{category_list}', _CATEGORY_LIST)


class LLMFilterExtractor:
    """
    LLM-based filter extraction for complex queries.

    Usage:
        extractor = LLMFilterExtractor()
        result = extractor.extract("32 inch monitor wall mount")
        if result:
            filters = result.filters
    """

    def __init__(self, model: str = None, temperature: float = 0.1):
        """
        Initialize LLM filter extractor.

        Args:
            model: OpenAI model to use (default: from env or gpt-4o-mini)
            temperature: Model temperature (lower = more deterministic)
        """
        self.model = model or os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano')
        self.temperature = temperature
        self.retry_config = RetryConfig(
            max_attempts=2,
            base_delay=0.5,
            max_delay=5.0
        )

    def extract(self, query: str, previous_category: str = None) -> Optional[LLMExtractionResult]:
        """
        Extract filters from query using LLM.

        Args:
            query: User's natural language query
            previous_category: Category from the user's previous search (for context)

        Returns:
            LLMExtractionResult with extracted filters, or None if LLM unavailable
        """
        client = get_openai_client()
        if not client:
            return None

        try:
            handler = RetryHandler(
                config=self.retry_config,
                operation_name="llm_filter_extraction"
            )

            result = handler.execute(
                lambda: self._call_openai(client, query, previous_category),
                fallback=None
            )

            if result is None:
                return None

            return self._parse_response(result, query)

        except Exception as e:
            # Log but don't fail - this is a fallback system
            _logger.error("LLM extraction error", extra={"error": str(e), "query": query})
            return None

    def _call_openai(self, client, query: str, previous_category: str = None) -> str:
        """Make OpenAI API call."""
        user_content = f"Query: {query}"

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=self.temperature,
            max_completion_tokens=500
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str, query: str) -> Optional[LLMExtractionResult]:
        """Parse LLM response into structured filters."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            # Build SearchFilters from LLM response
            filters = SearchFilters()

            # Map category
            if data.get('product_category'):
                filters.product_category = self._normalize_category(data['product_category'])

            # Map connectors
            if data.get('connector_from'):
                filters.connector_from = self._normalize_connector(data['connector_from'])
            if data.get('connector_to'):
                filters.connector_to = self._normalize_connector(data['connector_to'])

            # Map length (only if it's actually a cable length, not screen size)
            if data.get('length_ft') is not None:
                filters.length = float(data['length_ft'])
                filters.length_unit = 'ft'  # LLM always extracts in feet

            # Map length range (for "between X and Y" queries)
            if data.get('length_min_ft') is not None:
                filters.length_min = float(data['length_min_ft'])
                filters.length_unit = 'ft'
            if data.get('length_max_ft') is not None:
                filters.length_max = float(data['length_max_ft'])
                filters.length_unit = 'ft'

            # Map length preference
            if data.get('length_preference'):
                pref = data['length_preference'].lower()
                if 'longer' in pref:
                    filters.length_preference = LengthPreference.EXACT_OR_LONGER
                elif 'shorter' in pref:
                    filters.length_preference = LengthPreference.EXACT_OR_SHORTER

            # Map color
            if data.get('color'):
                filters.color = data['color'].lower()

            # Map features
            if data.get('features'):
                filters.features = data['features']

            # Map keywords to appropriate filter fields
            if data.get('keywords'):
                filters.keywords = data['keywords']

            # Map min_monitors
            if data.get('min_monitors'):
                filters.min_monitors = int(data['min_monitors'])

            # Map port count (store as feature for now)
            if data.get('port_count'):
                filters.port_count = int(data['port_count'])

            # Map network speed (for ethernet/network cables)
            if data.get('network_speed_mbps'):
                filters.requested_network_speed = int(data['network_speed_mbps'])

            # Map thunderbolt version (3 or 4)
            if data.get('thunderbolt_version'):
                try:
                    filters.thunderbolt_version = int(data['thunderbolt_version'])
                except (ValueError, TypeError):
                    pass

            # Map rack height (U-height for rack products)
            if data.get('rack_height'):
                try:
                    filters.rack_height = int(data['rack_height'])
                except (ValueError, TypeError):
                    pass

            # Map KVM video type (for KVM switches/extenders)
            if data.get('kvm_video_type'):
                filters.kvm_video_type = data['kvm_video_type']

            # Map power wattage (for docks, chargers)
            if data.get('requested_power_wattage'):
                try:
                    filters.requested_power_wattage = int(data['requested_power_wattage'])
                except (ValueError, TypeError):
                    pass

            # Map refresh rate (for monitors, cables, docks)
            if data.get('requested_refresh_rate'):
                try:
                    filters.requested_refresh_rate = int(data['requested_refresh_rate'])
                except (ValueError, TypeError):
                    pass

            # Map bay count (for storage enclosures)
            if data.get('bay_count'):
                try:
                    filters.bay_count = int(data['bay_count'])
                except (ValueError, TypeError):
                    pass

            # Map screen size (for privacy screens)
            if data.get('screen_size'):
                try:
                    filters.screen_size = float(data['screen_size'])
                except (ValueError, TypeError):
                    pass

            # Map drive size (for storage enclosures)
            if data.get('drive_size'):
                filters.drive_size = str(data['drive_size']).strip()

            # Map USB version (for hubs, adapters, docks)
            if data.get('usb_version'):
                filters.usb_version = str(data['usb_version']).strip()

            # Regex fallback for rack_height if LLM didn't extract it
            if not filters.rack_height and filters.product_category == 'rack':
                height_match = re.search(r'\b(\d+)\s*[Uu]\b', query)
                if height_match:
                    filters.rack_height = int(height_match.group(1))

            # Deduplicate: remove keywords that match structured filter values
            if filters.keywords:
                dedup_values = set()
                for val in [filters.usb_version, filters.connector_from, filters.connector_to,
                            filters.drive_size, filters.kvm_video_type, filters.cable_type]:
                    if val:
                        dedup_values.add(val.lower())
                if filters.thunderbolt_version:
                    dedup_values.update([f'thunderbolt {filters.thunderbolt_version}',
                                         f'tb{filters.thunderbolt_version}'])
                if dedup_values:
                    filters.keywords = [kw for kw in filters.keywords
                                        if kw.lower() not in dedup_values]

            return LLMExtractionResult(
                filters=filters,
                confidence=data.get('confidence', 0.8),
                reasoning=data.get('reasoning', '')
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            _logger.error("Error parsing LLM response", extra={"error": str(e)})
            return None

    def _normalize_category(self, category: str) -> str:
        """Normalize category name to match product metadata.

        Delegates to core.category_config for single source of truth.
        """
        from core.category_config import normalize_llm_category
        return normalize_llm_category(category)

    def _normalize_connector(self, connector: str) -> str:
        """Normalize connector name.

        Delegates to core.normalization for single source of truth.
        """
        from core.normalization import normalize_connector
        return normalize_connector(connector)



# Convenience function for one-off extraction
def extract_filters_with_llm(query: str) -> Optional[SearchFilters]:
    """
    Extract filters using LLM only (no rule-based).

    Useful for testing or when rule-based completely fails.

    Args:
        query: User's query

    Returns:
        SearchFilters or None if LLM unavailable
    """
    extractor = LLMFilterExtractor()
    result = extractor.extract(query)
    return result.filters if result else None


# Test function
if __name__ == "__main__":
    # Test queries that are problematic for rule-based extraction
    test_queries = [
        "32 inch monitor wall mount",
        "RAID enclosure for 4 drives",
        "7-port USB 3.0 hub",
        "DisplayPort to VGA adapter",
        "Thunderbolt 3 dock for dual 4K monitors",
        "wall mount server rack",
        "Cat6a ethernet cable 10ft",
    ]

    extractor = LLMFilterExtractor()

    print("LLM Filter Extraction Test")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = extractor.extract(query)

        if result:
            print(f"  Category: {result.filters.product_category}")
            print(f"  Connectors: {result.filters.connector_from} -> {result.filters.connector_to}")
            print(f"  Length: {result.filters.length}")
            print(f"  Features: {result.filters.features}")
            print(f"  Keywords: {getattr(result.filters, 'keywords', None)}")
            print(f"  Min Monitors: {result.filters.min_monitors}")
            print(f"  Confidence: {result.confidence}")
            print(f"  Reasoning: {result.reasoning}")
        else:
            print("  LLM extraction failed or unavailable")
