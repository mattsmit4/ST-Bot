"""
Centralized column configuration per product category.

Three-tier system:
- Tier 1: 5-7 key columns shown in initial search results (quick overview)
- Tier 2: Meaningful specs for comparing products and highlighting differences
          (Tier 1 + category-specific specs, MINUS physical/packaging/warranty)
- Tier 3: Full detail including physical dimensions, packaging, and warranty
          (only shown in single-product detail views)

Tier 2 and Tier 3 are derived from the same 'tier2' list in each category config.
TIER3_FIELDS defines the physical/packaging/warranty fields that get filtered out
of Tier 2 at runtime by get_tier2_columns().

This is the SINGLE SOURCE OF TRUTH for which metadata fields are displayed
per product type. All formatters, LLM response generators, and followup
handlers should read from this config rather than hardcoding field picks.

Column names use the NORMALIZED names from data.loader.COLUMN_ALIASES
(the right-side values, e.g., 'interface_a', not 'Connector_A').

To update: edit the tier1/tier2 dicts below. Run verify script to see
which fields are available:
    python scripts/verify_category_columns.py
"""

from typing import Dict, List, Optional, TypedDict


class TierConfig(TypedDict):
    tier1: Dict[str, str]  # field_name → display_label
    tier2: Dict[str, str]  # field_name → display_label


# =============================================================================
# USER: Fill in tier1 (5-7 key columns) and tier2 (all relevant columns)
# for each category. Comments show field population % from actual data.
#
# MAPPING: Your Excel "ItemCategory" → Internal categories
# ─────────────────────────────────────────────────────────
# "Cables - Other"              → cable (318)
# "Computer Cards and Adapters" → computer_card (161)
# "Data Storage"                → storage_enclosure (76), other (30)
# "Digital Display Cables"      → cable (249)
# "Display and Video Adapters"  → adapter (102), cable (46)
# "Display Mounts"              → display_mount (98)
# "Docking Stations"            → dock (30)
# "Hubs"                        → hub (91)
# "KVM Switches"                → kvm_switch (107), cable (33), kvm_extender (8)
# "Legacy Category"             → other (46)
# "Multiport Adapters"          → multiport_adapter (35)
# "Network Cables"              → cable (1,279), fiber_cable (147)
# "Networking"                  → network (392), adapter (62), ethernet_switch (11)
# "No Category"                 → other (917)
# "Racks and Enclosures"        → rack (180)
# "USB and Thunderbolt Cables"  → cable (347)
# "Video Display Connectivity"  → video_switch (14), video_splitter (13), adapter (15), other (43)
# "Workstation Accessories"     → privacy_screen (86), cable (77), mount (13), other (126), adapter (10)
# =============================================================================

CATEGORY_COLUMNS: Dict[str, TierConfig] = {

    # =========================================================================
    # CABLE (2,349 products)
    # From: "Network Cables", "USB and Thunderbolt Cables", "Cables - Other",
    #       "Digital Display Cables", "Workstation Accessories", "Display and
    #       Video Adapters", "KVM Switches"
    # =========================================================================
    'cable': {
        'tier1': {
            'cable_length_raw': 'Cable Length',
            'connector_style': 'Connector Style',
            'jacket_type': 'Cable Jacket Material',
            'wire_gauge': 'Wire Gauge',
            'connector_plating': 'Connector Plating',
            'Cable_USB_Host': 'Cable USB Host',
            'interface_a': 'Connector A',
            'interface_b': 'Connector B',
            # From Digital Display Cables
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'connectors_desc': 'Connectors',
        },
        'tier2': {
            'cable_length_raw': 'Cable Length',
            'connector_style': 'Connector Style',
            'jacket_type': 'Cable Jacket Material',
            'wire_gauge': 'Wire Gauge',
            'connector_plating': 'Connector Plating',
            'Cable_USB_Host': 'Cable USB Host',
            'interface_a': 'Connector A',
            'interface_b': 'Connector B',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'connectors_desc': 'Connectors',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            'active_passive': 'Active or Passive Adapter',
            'Cable_USB_Speed': 'Cable USB Speed',
            'Cable_USB_Per': 'Cable USB Per',
            'Cable_USB_Connection': 'Cable USB Connection',
            'shield_type': 'Cable Shield Material',
            'type_and_rate': 'Type and Rate',
            'power_delivery': 'Power Delivery',
            'Cable_OD': 'Cable OD',
            'Input_Current': 'Input Current',
            'MTBF': 'MTBF',
            'network_speed': 'Maximum Data Transfer Rate',
            'power_adapter': 'Power Source',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'Video_Revision': 'Video Revision',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'Input_Voltage': 'Input Voltage',
            'Supported_Protocols': 'Supported Protocols',
            'audio_ports': 'Audio Specifications',
            'Output_Current': 'Output Current',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'Output_Voltage': 'Output Voltage',
            'Plug_Type': 'Plug Type',
            'power_consumption': 'Power Consumption (In Watts)',
            'chipset': 'Chipset ID',
            'hub_ports_raw': 'Ports',
            'Number_of_Conductors': 'Number of Conductors',
            'conductor_type': 'Conductor Type',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'os_compatibility': 'OS Compatibility',
            'General_Specifications': 'General Specifications',
            'Insertion_Rating': 'Insertion Rating',
            'max_resolution': 'Maximum Analog Resolutions',
            # Retained from earlier categories
            'fire_rating': 'Fire Rating',
            'Number_of_Ferrites': 'Number of Ferrites',
            'standards': 'Industry Standards',
            'Impedance': 'Impedance',
            'Regulatory_Approvals': 'Regulatory Approvals',
        },
    },

    # =========================================================================
    # CABLE_ORGANIZER (77 products) - Cable clips, ties, fasteners, organizers
    # From: "Workstation Accessories" > "Cable Organizers and Fasteners" (77)
    # =========================================================================
    'cable_organizer': {
        'tier1': {
            'material': 'Material',
            'color': 'Color',
            'Package_Quantity': 'Package Quantity',
            'product_weight': 'Weight of Product',
            'General_Specifications': 'General Specifications',
        },
        'tier2': {
            'material': 'Material',
            'color': 'Color',
            'Package_Quantity': 'Package Quantity',
            'product_weight': 'Weight of Product',
            'General_Specifications': 'General Specifications',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            # Other
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'mount_options': 'Mounting Options',
        },
    },

    # =========================================================================
    # ADAPTER (189 products)
    # From: "Display and Video Adapters" (102), "Networking" (62),
    #       "Video Display Connectivity" (15), "Workstation Accessories" (10)
    # =========================================================================
    'adapter': {
        'tier1': {
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'dock_num_displays': 'Displays Supported',
            'av_input': 'AV Input',
            'av_output': 'AV Output',
            'cable_length_raw': 'Cable Length',
            'audio_ports': 'Audio Specifications',
        },
        'tier2': {
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'dock_num_displays': 'Displays Supported',
            'av_input': 'AV Input',
            'av_output': 'AV Output',
            'cable_length_raw': 'Cable Length',
            'audio_ports': 'Audio Specifications',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            # General
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            # Environmental
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            # Video / Display
            'Wide_Screen_Supported': 'Wide Screen Supported',
            'chipset': 'Chipset ID',
            'supported_resolutions': 'Supported Resolutions',
            'active_passive': 'Active or Passive Adapter',
            'kvm_audio': 'Audio',
            'power_delivery': 'Power Delivery',
            'hub_ports_raw': 'Ports',
            'Video_Revision': 'Video Revision',
            'Maximum_Cable_Distance_To_Display': 'Maximum Cable Distance To Display',
            'max_resolution': 'Maximum Analog Resolutions',
            'dock_4k_support': '4K Support',
            'Laptop_Charging_via_Power_Delivery': 'Laptop Charging via Power Delivery',
            'USB_Pass-Through': 'USB Pass-Through',
            'Output_Connectors': 'Output Connectors',
            'Input_Connectors': 'Input Connector(s)',
            'Memory': 'Memory',
            # Compatibility / Standards
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'OS_Compatibility': 'OS Compatibility',
            'MTBF': 'MTBF',
            'standards': 'Industry Standards',
            'Microsoft_WHQL_Certified': 'Microsoft WHQL Certified',
            # Power
            'power_adapter': 'Power Source',
            'Input_Voltage': 'Input Voltage',
            'Output_Voltage': 'Output Voltage',
            'power_consumption': 'Power Consumption (In Watts)',
            # Cable / Connector
            'jacket_type': 'Cable Jacket Material',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'Bandwidth': 'Bandwidth',
            'Accent_Color': 'Accent Color',
            'connector_plating': 'Connector Plating',
            'General_Specifications': 'General Specifications',
            'interface_a': 'Connector A',
            'interface_b': 'Connector B',
            'fire_rating': 'Fire Rating',
            'usb_type': 'Bus Type',
            'LED_Indicators': 'LED Indicators',
        },
    },

    # =========================================================================
    # NETWORK (392 products) - SFP modules, media converters, network cards
    # From: "Networking" (392)
    # =========================================================================
    'network': {
        'tier1': {
            'nw_cable_type': 'Compatible Networks',
            'network_speed': 'Maximum Data Transfer Rate',
            'standards': 'Industry Standards',
            'Compatible_Brand': 'Compatible Brand',
            'power_consumption': 'Power Consumption (In Watts)',
            'fiber_type_raw': 'Fiber Type',
            'Max_Transfer_Distance': 'Max Transfer Distance',
        },
        'tier2': {
            'nw_cable_type': 'Compatible Networks',
            'network_speed': 'Maximum Data Transfer Rate',
            'standards': 'Industry Standards',
            'Compatible_Brand': 'Compatible Brand',
            'power_consumption': 'Power Consumption (In Watts)',
            'fiber_type_raw': 'Fiber Type',
            'Max_Transfer_Distance': 'Max Transfer Distance',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'storage_temp': 'Storage Temperature',
            'humidity': 'Humidity',
            'MTBF': 'MTBF',
            'Fiber_Connector': 'Fiber Connector',
            'DDM': 'DDM',
            'fiber_duplex_raw': 'Fiber Operating Mode',
            'fiber_wavelength_raw': 'Wavelength',
            'Output_Voltage': 'Output Voltage',
            'Auto_MDIX': 'Auto MDIX',
            'chipset': 'Chipset ID',
            'os_compatibility': 'OS Compatibility',
            'hub_ports_raw': 'Ports',
            'cable_length_raw': 'Cable Length',
            'power_adapter': 'Power Source',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'Jumbo_Frame_Support': 'Jumbo Frame Support',
            'Full_Duplex_Support': 'Full Duplex Support',
            'usb_type': 'Bus Type',
            'Input_Voltage': 'Input Voltage',
            'io_interface_raw': 'Interface',
            'jacket_type': 'Cable Jacket Material',
            'Wake_On_Lan': 'Wake On Lan',
            'Impedance': 'Impedance',
            'wire_gauge': 'Wire Gauge',
            'Local_Unit_Connectors': 'Local Unit Connectors',
            'WDM': 'WDM',
            'Promiscuous_Mode': 'Promiscuous Mode',
            'Plug_Type': 'Plug Type',
            'PXE': 'PXE',
            'Output_Current': 'Output Current',
            'Input_Current': 'Input Current',
            'form_factor': 'Form Factor',
            'USB_Ports_Supported': 'USB Ports Supported',
            'poe': 'PoE',
            'General_Specifications': 'General Specifications',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'Supported_Protocols': 'Supported Protocols',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'type_and_rate': 'Type and Rate',
            'max_distance': 'Max Distance',
            'Flow_Control': 'Flow Control',
            'external_ports': 'External Ports',
            'Switching_Architecture': 'Switching Architecture',
            'LED_Indicators': 'LED Indicators',
            'Port_Style': 'Port Style',
            'Antenna_Configuration': 'Antenna Configuration',
            'Buffer_Size': 'Buffer Size',
            'conn_type': 'Connector Type(s)',
            'Surge_Protection': 'Surge Protection',
            'USB_Pass-Through': 'USB Pass-Through',
            'Frequency': 'Frequency',
            'Frequency_Range': 'Frequency Range',
            'interface_a': 'Connector A',
            'Remote_Management_Ability': 'Remote Management Ability',
            'Bandwidth': 'Bandwidth',
            'Auto-Negotiation': 'Auto-Negotiation',
            'interface_b': 'Connector B',
            'Cabling': 'Cabling',
            'Internal_Ports': 'Internal Ports',
            'Rack-Mountable': 'Rack-Mountable',
            'Anti-Theft': 'Anti-Theft',
            'Cable_Management': 'Cable Management',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'power_delivery': 'Power Delivery',
            'Fans': 'Fan(s)',
            'Front_Door_Construction': 'Front Door Construction',
            'Front_Door_Features': 'Front Door Features',
            'Height_Adjustment': 'Height Adjustment',
            'Wallmountable': 'Wallmountable',
            'Tools_Included': 'Tools Included',
            'Regulatory_Approvals': 'Regulatory Approvals',
            'Compatible_Lock_Slot': 'Compatible Lock Slot',
            'UASP_Support': 'UASP Support',
            'USB-C_Device_Ports': 'USB-C Device Port(s)',
            'USB-C_Host_Connection': 'USB-C Host Connection',
            'mount_options': 'Mounting Options',
        },
    },

    # =========================================================================
    # COMPUTER_CARD (161 products) - PCIe cards, serial cards, USB cards
    # From: "Computer Cards and Adapters" (161)
    # =========================================================================
    'computer_card': {
        'tier1': {
            'hub_ports_raw': 'Ports',
            'io_interface_raw': 'Interface',
            'usb_type': 'Bus Type',
            'chipset': 'Chipset ID',
            'Port_Style': 'Port Style',
            'power_adapter': 'Power Source',
        },
        'tier2': {
            'hub_ports_raw': 'Ports',
            'io_interface_raw': 'Interface',
            'usb_type': 'Bus Type',
            'chipset': 'Chipset ID',
            'Port_Style': 'Port Style',
            'power_adapter': 'Power Source',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            # General
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            # Environmental
            'operating_temp': 'Operating Temperature',
            'storage_temp': 'Storage Temperature',
            'humidity': 'Humidity',
            # Compatibility
            'OS_Compatibility': 'OS Compatibility',
            # Electrical / Power
            'Input_Voltage': 'Input Voltage',
            'Output_Voltage': 'Output Voltage',
            'Input_Current': 'Input Current',
            'Output_Current': 'Output Current',
            'power_consumption': 'Power Consumption (In Watts)',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'Total_USB_Power_Output': 'Total USB Power Output',
            'Surge_Protection': 'Surge Protection',
            'Isolation': 'Isolation',
            'Input_Voltage_Frequency': 'Input Voltage Frequency',
            # Serial / Communication
            'Max_Baud_Rate': 'Max Baud Rate',
            'Serial_Protocol': 'Serial Protocol',
            'Data_Bits': 'Data Bits',
            'Parity': 'Parity',
            'FIFO': 'FIFO',
            'Stop_Bits': 'Stop Bits',
            'Flow_Control': 'Flow Control',
            'Full_Duplex_Support': 'Full Duplex Support',
            'Auto_MDIX': 'Auto MDIX',
            'Auto-Negotiation': 'Auto-Negotiation',
            # Data / Transfer
            'network_speed': 'Maximum Data Transfer Rate',
            'type_and_rate': 'Type and Rate',
            'UASP_Support': 'UASP Support',
            'Supported_Protocols': 'Supported Protocols',
            'Bandwidth': 'Bandwidth',
            'Buffer_Size': 'Buffer Size',
            # Cable / Connector
            'cable_length_raw': 'Cable Length',
            'interface_a': 'Connector A',
            'interface_b': 'Connector B',
            'Plug_Type': 'Plug Type',
            'conn_type': 'Connector Type(s)',
            'host_connector': 'Host Connectors',
            'external_ports': 'External Ports',
            'Internal_Ports': 'Internal Ports',
            'connector_plating': 'Connector Plating',
            'connector_style': 'Connector Style',
            'jacket_type': 'Cable Jacket Material',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'Cable_OD': 'Cable OD',
            'shield_type': 'Cable Shield Material',
            'wire_gauge': 'Wire Gauge',
            'Number_of_Conductors': 'Number of Conductors',
            'Cabling': 'Cabling',
            'Thread_Types': 'Thread Type(s)',
            # Standards / Compliance
            'standards': 'Industry Standards',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'MTBF': 'MTBF',
            'Microsoft_WHQL_Certified': 'Microsoft WHQL Certified',
            'Regulatory_Approvals': 'Regulatory Approvals',
            # Video / Display
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'max_resolution': 'Maximum Analog Resolutions',
            'supported_resolutions': 'Supported Resolutions',
            'dock_num_displays': 'Displays Supported',
            'av_input': 'AV Input',
            'av_output': 'AV Output',
            # Networking
            'nw_cable_type': 'Compatible Networks',
            'audio_ports': 'Audio Specifications',
            'Remote_Management_Ability': 'Remote Management Ability',
            'Max_Transfer_Distance': 'Max Transfer Distance',
            'Jumbo_Frame_Support': 'Jumbo Frame Support',
            'Wake_On_Lan': 'Wake On Lan',
            # Other
            'General_Specifications': 'General Specifications',
            'kvm_audio': 'Audio',
            'power_delivery': 'Power Delivery',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'LED_Indicators': 'LED Indicators',
            'Accent_Color': 'Accent Color',
            # Fan / Cooling
            'Air_Flow_Rate': 'Air Flow Rate',
            'Fan_Bearing_Type': 'Fan Bearing Type',
            'Noise_Level': 'Noise Level',
            'Fan_RPM': 'Fan RPM',
            'Fans': 'Fan(s)',
            'Fan_Size': 'Fan Size',
            # Specialty
            'Head_Pattern': 'Head Pattern',
            'Evaporation': 'Evaporation',
            'Specific_Gravity': 'Specific Gravity',
            'Bleed': 'Bleed',
            'Internal_Depth': 'Internal Depth',
            'Internal_Length': 'Internal Length',
        },
    },

    # =========================================================================
    # DISPLAY_MOUNT (98 products) - Monitor mounts, TV mounts, desk arms
    # From: "Display Mounts" (98)
    # =========================================================================
    'display_mount': {
        'tier1': {
            'mount_options': 'Mounting Options',
            'dock_num_displays': 'Displays Supported',
            'mount_max_display_size': 'Maximum Display Size',
            'Wallmountable': 'Wallmountable',
            'Display_Rotation': 'Display Rotation',
            'mount_min_display_size': 'Minimum Display Size',
        },
        'tier2': {
            'mount_options': 'Mounting Options',
            'dock_num_displays': 'Displays Supported',
            'mount_max_display_size': 'Maximum Display Size',
            'Wallmountable': 'Wallmountable',
            'Display_Rotation': 'Display Rotation',
            'mount_min_display_size': 'Minimum Display Size',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            # General
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            # Environmental
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            # Mount adjustments
            'Display_Tilt': 'Display Tilt',
            'Articulating': 'Articulating',
            'Height_Adjustment': 'Height Adjustment',
            'Swivel_/_Pivot': 'Swivel / Pivot',
            'Cable_Management': 'Cable Management',
            'mount_curved_tv': 'Fits Curved Display',
            'Maximum_Arm_Extension': 'Maximum Arm Extension',
            'Mounting_Surface_Thickness': 'Mounting Surface Thickness',
            'Display_Swivel': 'Display Swivel',
            'Maximum_Height': 'Maximum Height',
            'Fits_UltraWide_Displays': 'Fits UltraWide Displays',
            'Anti-Theft': 'Anti-Theft',
            'Minimum_Height': 'Minimum Height',
            'mount_material': 'Construction Material',
            'Video_Wall': 'Video Wall',
            'Tools_Included': 'Tools Included',
            'Minimum_Arm_Extension': 'Minimum Arm Extension',
            'General_Specifications': 'General Specifications',
            'Minimum_Profile_from_Wall': 'Minimum Profile from Wall',
            'Orientation': 'Orientation',
            'Maximum_Arm_Span': 'Maximum Arm Span',
            # Tablet / Laptop
            'Maximum_Tablet_Thickness': 'Maximum Tablet Thickness',
            'Laptop_Tray_Height': 'Laptop Tray Height',
            'Laptop_Tray_Tilt': 'Laptop Tray Tilt',
            'Laptop_Tray_Width': 'Laptop Tray Width',
            'Aspect_Ratio': 'Aspect Ratio',
            'Laptop_Arm_Extension': 'Laptop Arm Extension',
            # VESA / Dimensions
            'mount_vesa_pattern': 'VESA Hole Pattern(s)',
            'Internal_Height': 'Internal Height',
            'Internal_Width': 'Internal Width',
            'Weight_Capacity_of_Work_Surface': 'Weight Capacity of Work Surface',
            'Flat_Pack_(Assembly_Required)': 'Flat Pack (Assembly Required)',
            'Internal_Depth': 'Internal Depth',
        },
    },

    # =========================================================================
    # DOCK (30 products) - Docking stations
    # From: "Docking Stations" (30)
    # =========================================================================
    'dock': {
        'tier1': {
            'dock_num_displays': 'Displays Supported',
            'dock_4k_support': '4K Support',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'io_interface_raw': 'Interface',
            'total_ports': 'Total Ports',
        },
        'tier2': {
            'dock_num_displays': 'Displays Supported',
            'dock_4k_support': '4K Support',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'USB_Ports_Supported': 'USB Ports Supported',
            'io_interface_raw': 'Interface',
            'total_ports': 'Total Ports',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            # General
            'warranty': 'Warranty',
            'material': 'Material',
            'color': 'Color',
            # Environmental
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            # Power (user-facing specs only; raw electrical details omitted)
            'power_adapter': 'Power Source',
            'power_consumption': 'Power Consumption (In Watts)',
            'power_delivery': 'Power Delivery',
            'Supported_Charging_Outputs': 'Supported Charging Outputs',
            # Compatibility
            'OS_Compatibility': 'OS Compatibility',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            # Networking
            'nw_cable_type': 'Compatible Networks',
            'PXE': 'PXE',
            'Wake_On_Lan': 'Wake On Lan',
            'Full_Duplex_Support': 'Full Duplex Support',
            'Auto_MDIX': 'Auto MDIX',
            'network_speed': 'Maximum Data Transfer Rate',
            # Data / Transfer
            'type_and_rate': 'Type and Rate',
            'kvm_audio': 'Audio',
            'UASP_Support': 'UASP Support',
            'MTBF': 'MTBF',
            'standards': 'Industry Standards',
            # Connector / Ports
            'cable_length_raw': 'Cable Length',
            'host_connector': 'Host Connectors',
            'chipset': 'Chipset ID',
            'Compatible_Lock_Slot': 'Compatible Lock Slot',
            'audio_ports': 'Audio Specifications',
            'USB_Pass-Through': 'USB Pass-Through',
            # Other
            'LED_Indicators': 'LED Indicators',
            'General_Specifications': 'General Specifications',
            'Memory_Media_Type': 'Memory Media Type',
            'max_resolution': 'Maximum Analog Resolutions',
            'Supported_Protocols': 'Supported Protocols',
        },
    },

    # =========================================================================
    # HUB (91 products) - USB hubs
    # From: "Hubs" (91)
    # =========================================================================
    'hub': {
        'tier1': {
            'hub_ports_raw': 'Ports',
            'io_interface_raw': 'Interface',
            'power_adapter': 'Power Source',
            'USB-C_Host_Connection': 'USB-C Host Connection',
        },
        'tier2': {
            'hub_ports_raw': 'Ports',
            'network_speed': 'Maximum Data Transfer Rate',
            'power_adapter': 'Power Source',
            'io_interface_raw': 'Interface',
            'USB-C_Host_Connection': 'USB-C Host Connection',
            'type_and_rate': 'Type and Rate',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            # General
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            # Environmental
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            # Compatibility
            'OS_Compatibility': 'OS Compatibility',
            # Ports / USB
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'USB-C_Device_Ports': 'USB-C Device Port(s)',
            'usb_type': 'Bus Type',
            'chipset': 'Chipset ID',
            'form_factor': 'Form Factor',
            'USB_Ports_Supported': 'USB Ports Supported',
            'Total_USB_Power_Output': 'Total USB Power Output',
            'USB_Pass-Through': 'USB Pass-Through',
            'host_connector': 'Host Connectors',
            'interface_a': 'Connector A',
            'Port_Multiplier': 'Port Multiplier',
            # Power
            'Output_Voltage': 'Output Voltage',
            'Input_Voltage': 'Input Voltage',
            'power_consumption': 'Power Consumption (In Watts)',
            'power_delivery': 'Power Delivery',
            'Output_Current': 'Output Current',
            'Plug_Type': 'Plug Type',
            'Input_Current': 'Input Current',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'Surge_Protection': 'Surge Protection',
            'Supported_Charging_Outputs': 'Supported Charging Outputs',
            'Laptop_Charging_via_Power_Delivery': 'Laptop Charging via Power Delivery',
            # Cable
            'cable_length_raw': 'Cable Length',
            'jacket_type': 'Cable Jacket Material',
            # Standards / Compliance
            'standards': 'Industry Standards',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'UASP_Support': 'UASP Support',
            'MTBF': 'MTBF',
            # Indicators
            'LED_Indicators': 'LED Indicators',
            # Networking
            'Auto_MDIX': 'Auto MDIX',
            'Full_Duplex_Support': 'Full Duplex Support',
            'nw_cable_type': 'Compatible Networks',
            'Jumbo_Frame_Support': 'Jumbo Frame Support',
            'Wake_On_Lan': 'Wake On Lan',
            'Flow_Control': 'Flow Control',
            'PXE': 'PXE',
            'Promiscuous_Mode': 'Promiscuous Mode',
            # Serial
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'Data_Bits': 'Data Bits',
            'FIFO': 'FIFO',
            'Max_Baud_Rate': 'Max Baud Rate',
            'Parity': 'Parity',
            'Serial_Protocol': 'Serial Protocol',
            'Stop_Bits': 'Stop Bits',
            # Other
            'General_Specifications': 'General Specifications',
            'u_height': 'U Height',
            'Memory_Media_Type': 'Memory Media Type',
        },
    },

    # =========================================================================
    # MULTIPORT_ADAPTER (35 products)
    # From: "Multiport Adapters" (35)
    # =========================================================================
    'multiport_adapter': {
        'tier1': {
            'dock_num_displays': 'Displays Supported',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'io_interface_raw': 'Interface',
            'power_delivery': 'Power Delivery',
            'dock_4k_support': '4K Support',
            'type_and_rate': 'Type and Rate',
            'network_speed': 'Maximum Data Transfer Rate',
        },
        'tier2': {
            'dock_num_displays': 'Displays Supported',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'io_interface_raw': 'Interface',
            'power_delivery': 'Power Delivery',
            'dock_4k_support': '4K Support',
            'type_and_rate': 'Type and Rate',
            'network_speed': 'Maximum Data Transfer Rate',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            'usb_type': 'Bus Type',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'os_compatibility': 'OS Compatibility',
            'cable_length_raw': 'Cable Length',
            'power_adapter': 'Power Source',
            'USB_Ports_Supported': 'USB Ports Supported',
            'total_ports': 'Total Ports',
            'UASP_Support': 'UASP Support',
            'Output_Voltage': 'Output Voltage',
            'power_consumption': 'Power Consumption (In Watts)',
            'nw_cable_type': 'Compatible Networks',
            'Full_Duplex_Support': 'Full Duplex Support',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'Wake_On_Lan': 'Wake On Lan',
            'PXE': 'PXE',
            'standards': 'Industry Standards',
            'chipset': 'Chipset ID',
            'hub_ports_raw': 'Ports',
            'USB_Pass-Through': 'USB Pass-Through',
            'kvm_audio': 'Audio',
            'Auto_MDIX': 'Auto MDIX',
            'General_Specifications': 'General Specifications',
            'MTBF': 'MTBF',
            'LED_Indicators': 'LED Indicators',
            'audio_ports': 'Audio Specifications',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'max_resolution': 'Maximum Analog Resolutions',
            'av_input': 'AV Input',
            'av_output': 'AV Output',
            'form_factor': 'Form Factor',
            'Microsoft_WHQL_Certified': 'Microsoft WHQL Certified',
            'Wide_Screen_Supported': 'Wide Screen Supported',
            'supported_resolutions': 'Supported Resolutions',
            'USB-C_Host_Connection': 'USB-C Host Connection',
            'Memory': 'Memory',
        },
    },

    # =========================================================================
    # KVM_SWITCH (107 products)
    # From: "KVM Switches" (107)
    # =========================================================================
    'kvm_switch': {
        'tier1': {
            'kvm_ports_raw': 'KVM Ports',
            'kvm_audio': 'Audio',
            'kvm_interface': 'PC Interface',
            'IP_Control': 'IP Control',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'Cables_Included': 'Cables Included',
            'mount_num_displays': 'Number of Monitors Supported',
        },
        'tier2': {
            'kvm_ports_raw': 'KVM Ports',
            'kvm_audio': 'Audio',
            'kvm_interface': 'PC Interface',
            'IP_Control': 'IP Control',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'Cables_Included': 'Cables Included',
            'mount_num_displays': 'Number of Monitors Supported',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Included_in_Package': 'Included in Package',
            'Package_Quantity': 'Package Quantity',
            # General
            'color': 'Color',
            'material': 'Material',
            'warranty': 'Warranty',
            # Environmental
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            # KVM features
            'kvm_video_type': 'PC Video Type',
            'On-Screen_Display': 'On-Screen Display',
            'Rack-Mountable': 'Rack-Mountable',
            'Hot-Key_Selection': 'Hot-Key Selection',
            'Daisy-Chain': 'Daisy-Chain',
            'Push_Button_Hotkeys_and_Software': 'Push Button, Hotkeys, and Software',
            'Auto_Scan': 'Auto Scan',
            'Maximum_Number_of_Users': 'Maximum Number of Users',
            'Maximum_Cascaded_Computers': 'Maximum Cascaded Computers',
            'DVI_Support': 'DVI Support',
            # Compatibility
            'OS_Compatibility': 'OS Compatibility',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            # Power
            'power_adapter': 'Power Source',
            'Input_Voltage': 'Input Voltage',
            'Output_Voltage': 'Output Voltage',
            'Input_Current': 'Input Current',
            'power_consumption': 'Power Consumption (In Watts)',
            'Output_Current': 'Output Current',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'Plug_Type': 'Plug Type',
            # Cable / Connector
            'cable_length_raw': 'Cable Length',
            'audio_ports': 'Audio Specifications',
            'host_connector': 'Host Connectors',
            'interface_a': 'Connector A',
            'interface_b': 'Connector B',
            'connector_plating': 'Connector Plating',
            'jacket_type': 'Cable Jacket Material',
            'shield_type': 'Cable Shield Material',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            # Video / Display
            'Wide_Screen_Supported': 'Wide Screen Supported',
            'Video_Revision': 'Video Revision',
            'supported_resolutions': 'Supported Resolutions',
            'max_resolution': 'Maximum Analog Resolutions',
            'Color_Depth': 'Color Depth',
            'Brightness': 'Brightness',
            'Contrast_Ratio': 'Contrast Ratio',
            'Display_Size': 'Display Size',
            'Aspect_Ratio': 'Aspect Ratio',
            'Viewing_Angle': 'Viewing Angle',
            # Reliability / Standards
            'MTBF': 'MTBF',
            'Regulatory_Approvals': 'Regulatory Approvals',
            'standards': 'Industry Standards',
            'Microsoft_WHQL_Certified': 'Microsoft WHQL Certified',
            'LED_Indicators': 'LED Indicators',
            # Networking / Data
            'network_speed': 'Maximum Data Transfer Rate',
            'Max_Transfer_Distance': 'Max Transfer Distance',
            'Auto-Negotiation': 'Auto-Negotiation',
            'nw_cable_type': 'Compatible Networks',
            'DDM': 'DDM',
            # Fiber
            'fiber_wavelength_raw': 'Wavelength',
            'Fiber_Connector': 'Fiber Connector',
            'fiber_duplex_raw': 'Fiber Operating Mode',
            'fiber_type_raw': 'Fiber Type',
            # Rack / Mount
            'u_height': 'U Height',
            'Product_Depth': 'Product Depth',
            'Maximum_Mounting_Depth': 'Maximum Mounting Depth',
            'Minimum_Mounting_Depth': 'Minimum Mounting Depth',
            'Front_Door_Key_Lock': 'Front Door Key Lock',
            # Other
            'chipset': 'Chipset ID',
            'active_passive': 'Active or Passive Adapter',
            'General_Specifications': 'General Specifications',
        },
    },

    # =========================================================================
    # KVM_EXTENDER (8 products)
    # From: "KVM Switches" (8) - KVM extenders subset
    # =========================================================================
    'kvm_extender': {
        'tier1': {
            'kvm_video_type': 'PC Video Type',
            'kvm_ports_raw': 'KVM Ports',
            'kvm_interface': 'PC Interface',
            'kvm_audio': 'Audio',
            'audio_ports': 'Audio Specifications',
        },
        'tier2': {
            'kvm_video_type': 'PC Video Type',
            'kvm_ports_raw': 'KVM Ports',
            'Cables_Included': 'Cables Included',
            'kvm_interface': 'PC Interface',
            'IP_Control': 'IP Control',
            'Rack-Mountable': 'Rack-Mountable',
            'kvm_audio': 'Audio',
            'audio_ports': 'Audio Specifications',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'features': 'Features',
            'color': 'Color',
            'Max_Transfer_Distance': 'Max Transfer Distance',
            'fiber_type_raw': 'Fiber Type',
            'Fiber_Connector': 'Fiber Connector',
            'fiber_wavelength_raw': 'Wavelength',
            'Serial_Protocol': 'Serial Protocol',
        },
    },

    # =========================================================================
    # FIBER_CABLE (147 products)
    # From: "Network Cables" (147) - fiber optic subset
    # =========================================================================
    'fiber_cable': {
        'tier1': {
            'cable_length_raw': 'Cable Length',
            'network_rating_raw': 'Cable Rating',
            'fiber_type': 'Fiber Type',
            'fiber_connector': 'Fiber Connector',
            'fiber_duplex': 'Duplex Mode',
            'interface_a': 'Connector A',
            'interface_b': 'Connector B',
        },
        'tier2': {
            'cable_length_raw': 'Cable Length',
            'network_rating_raw': 'Cable Rating',
            'Number_of_Conductors': 'Number of Conductors',
            'wire_gauge': 'Wire Gauge',
            'conductor_type': 'Conductor Type',
            'interface_a': 'Connector A',
            'interface_b': 'Connector B',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            'fire_rating': 'Fire Rating',
            'jacket_type': 'Cable Jacket Material',
            'Wiring_Standard': 'Wiring Standard',
            'connector_plating': 'Connector Plating',
            'Cable_OD': 'Cable OD',
            'network_speed': 'Maximum Data Transfer Rate',
            'ETL_Verified': 'ETL Verified',
            'connector_style': 'Connector Style',
            'shield_type': 'Cable Shield Material',
            'Fiber_Size': 'Fiber Size',
            'fiber_type_raw': 'Fiber Type',
            'Fiber_Classificiation': 'Fiber Classification',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'fiber_wavelength_raw': 'Wavelength',
            'active_passive': 'Active or Passive Adapter',
            'General_Specifications': 'General Specifications',
            'MTBF': 'MTBF',
            'Solid_Copper': 'Solid Copper',
            'power_delivery': 'Power Delivery',
            'host_connector': 'Host Connectors',
            'standards': 'Industry Standards',
            'hub_ports_raw': 'Ports',
            'Insertion_Rating': 'Insertion Rating',
            'conn_type': 'Connector Type(s)',
        },
    },

    # =========================================================================
    # STORAGE_ENCLOSURE (76 products)
    # From: "Data Storage" (76)
    # =========================================================================
    'storage_enclosure': {
        'tier1': {
            'num_drives_raw': 'Number of Drives',
            'io_interface_raw': 'Interface',
            'type_and_rate': 'Type and Rate',
            'drive_size_raw': 'Drive Size',
            'network_speed': 'Maximum Data Transfer Rate',
        },
        'tier2': {
            'num_drives_raw': 'Number of Drives',
            'io_interface_raw': 'Interface',
            'type_and_rate': 'Type and Rate',
            'drive_size_raw': 'Drive Size',
            'network_speed': 'Maximum Data Transfer Rate',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            # General
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            # Environmental
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            # Compatibility
            'OS_Compatibility': 'OS Compatibility',
            'drive_compatibility_raw': 'Compatible Drive Types',
            # Drive / Storage
            'chipset': 'Chipset ID',
            'usb_type': 'Bus Type',
            'Drive_Installation': 'Drive Installation',
            'drive_connector_raw': 'Drive Connectors',
            'Hardware_Raid_Supported': 'Hardware Raid Supported',
            'Hot_Swap_Capability': 'Hot Swap Capability',
            'TRIM_Support': 'TRIM Support',
            'UASP_Support': 'UASP Support',
            'Port_Multiplier': 'Port Multiplier',
            'S.M.A.R.T._Support': 'S.M.A.R.T. Support',
            'Max_Drive_Height': 'Max Drive Height',
            '4Kn_Support': '4Kn Support',
            'Bootable': 'Bootable',
            'ATAPI_Support': 'ATAPI Support',
            'LBA_Support': 'LBA Support',
            'Memory_Media_Type': 'Memory Media Type',
            'Duplication_Speed': 'Duplication Speed',
            'Shock_Protection': 'Shock Protection',
            # Electrical / Power
            'Output_Voltage': 'Output Voltage',
            'Input_Voltage': 'Input Voltage',
            'power_adapter': 'Power Source',
            'power_consumption': 'Power Consumption (In Watts)',
            'Output_Current': 'Output Current',
            'Input_Current': 'Input Current',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            # Reliability
            'MTBF': 'MTBF',
            'LED_Indicators': 'LED Indicators',
            'Insertion_Rating': 'Insertion Rating',
            'Temperature_Alarm': 'Temperature Alarm',
            # Fan / Cooling
            'Fans': 'Fan(s)',
            'Fan_Size': 'Fan Size',
            'Fan_Bearing_Type': 'Fan Bearing Type',
            'Noise_Level': 'Noise Level',
            'Air_Flow_Rate': 'Air Flow Rate',
            # Cable / Connector
            'cable_length_raw': 'Cable Length',
            'hub_ports_raw': 'Ports',
            'host_connector': 'Host Connectors',
            'Plug_Type': 'Plug Type',
            'conn_type': 'Connector Type(s)',
            'Port_Style': 'Port Style',
            # Enclosure / Bays
            'Front_Door_Key_Lock': 'Front Door Key Lock',
            'Number_of_2.5_Inch_Bays': 'Number of 2.5 Inch Bays',
            'Number_of_3.5_Inch_Bays': 'Number of 3.5 Inch Bays',
            'Number_of_External_5.25_Inch_Bays': 'Number of External 5.25 Inch Bays',
            # Standards / Other
            'General_Specifications': 'General Specifications',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'standards': 'Industry Standards',
            'Buffer_Size': 'Buffer Size',
            'Wake_On_Lan': 'Wake On Lan',
        },
    },

    # =========================================================================
    # RACK (180 products) - Server racks, rack shelves, PDUs
    # From: "Racks and Enclosures" (180)
    # =========================================================================
    'rack': {
        'tier1': {
            'u_height': 'U Height',
            'Frame_Type': 'Frame Type',
            'standards': 'Industry Standards',
            'Wallmountable': 'Wallmountable',
            'rack_type': 'Rack Type',
            'Mounting_Rail_Profile': 'Mounting Rail Profile',
            'Cable_Management': 'Cable Management',
            'Front_Door_Features': 'Front Door Features',
            'Rear_Door_Features': 'Rear Door Features',
            # PDU fields (None for server racks, skipped by serializer)
            'Number_of_Power_Outlets': 'Number of Outlets',
            'Input_Voltage': 'Input Voltage',
            'Maximum_Output_Voltage': 'Maximum Output Voltage',
            'Load_Capacity': 'Load Capacity',
            'Input_Connectors': 'Input Connectors',
            'Output_Connectors': 'Output Connectors',
        },
        'tier2': {
            'u_height': 'U Height',
            'Frame_Type': 'Frame Type',
            'standards': 'Industry Standards',
            'Wallmountable': 'Wallmountable',
            'rack_type': 'Rack Type',
            'Mounting_Rail_Profile': 'Mounting Rail Profile',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'Maximum_Mounting_Depth': 'Maximum Mounting Depth',
            'Flat_Pack_Assembly_Required': 'Flat Pack (Assembly Required)',
            'Mounting_Hole_Types': 'Mounting Hole Types',
            'mount_material': 'Construction Material',
            'Weight_Capacity': 'Weight Capacity',
            'Internal_Width': 'Internal Width',
            'Cable_Management': 'Cable Management',
            'Minimum_Mounting_Depth': 'Minimum Mounting Depth',
            'Internal_Depth': 'Internal Depth',
            'Included': 'Included',
            'Thread_Types': 'Thread Type(s)',
            'Side_Panel_Construction': 'Side Panel Construction',
            'Front_Door_Construction': 'Front Door Construction',
            'Front_Door_Features': 'Front Door Features',
            'Side_Panel_Features': 'Side Panel Features',
            'Product_Height_with_Casters': 'Product Height with Casters',
            'mount_options': 'Mounting Options',
            'Rear_Door_Construction': 'Rear Door Construction',
            'Rear_Door_Features': 'Rear Door Features',
            'Orientation': 'Orientation',
            'Input_Voltage': 'Input Voltage',
            'Maximum_Total_Current_Draw_per_Phase': 'Maximum Total Current Draw per Phase',
            'Maximum_Total_Current': 'Maximum Total Current',
            'form_factor': 'Form Factor',
            'Fan_Options': 'Fan Options',
            'Regulatory_Derated_Input_Current': 'Regulatory Derated Input Current',
            'Rack-Mountable': 'Rack-Mountable',
            'Number_of_Power_Outlets': 'Number of Power Outlets',
            'Maximum_Output_Voltage': 'Maximum Output Voltage',
            'Nominal_Input_Voltage': 'Nominal Input Voltage',
            'Nominal_Output_Voltage': 'Nominal Output Voltage',
            'Weight_Capacity_for_Racks_Rolling': 'Weight Capacity for Racks (Rolling)',
            'General_Specifications': 'General Specifications',
            'Load_Capacity': 'Load Capacity',
            'operating_temp': 'Operating Temperature',
            'storage_temp': 'Storage Temperature',
            'cable_length_raw': 'Cable Length',
            'Rail_Thickness': 'Rail Thickness',
            'humidity': 'Humidity',
            'Air_Flow_Rate': 'Air Flow Rate',
            'Tools_Included': 'Tools Included',
            'Weight_Capacity_of_Work_Surface': 'Weight Capacity of Work Surface',
            'LED_Indicators': 'LED Indicators',
            'Mounting_Surface_Thickness': 'Mounting Surface Thickness',
            'Plug_Type': 'Plug Type',
            'Internal_Height': 'Internal Height',
            'power_adapter': 'Power Source',
            'Internal_Length': 'Internal Length',
            'Output_Voltage': 'Output Voltage',
            'Fan_Bearing_Type': 'Fan Bearing Type',
            'Input_Current': 'Input Current',
            'Input_Voltage_Frequency': 'Input Voltage Frequency',
            'Output_Current': 'Output Current',
            'Fan_RPM': 'Fan RPM',
            'MTBF': 'MTBF',
            'power_consumption': 'Power Consumption (In Watts)',
            'Noise_Level': 'Noise Level',
        },
    },

    # =========================================================================
    # PRIVACY_SCREEN (86 products)
    # From: "Workstation Accessories" (86)
    # =========================================================================
    'privacy_screen': {
        'tier1': {
            'Screen_Size': 'Screen Size',
            'Aspect_Ratio': 'Aspect Ratio',
            'Orientation': 'Orientation',
            'mount_options': 'Mounting Options',
        },
        'tier2': {
            'mount_options': 'Mounting Options',
            'Screen_Size': 'Screen Size',
            'Aspect_Ratio': 'Aspect Ratio',
            'Orientation': 'Orientation',
            'Output_Current': 'Output Current',
            'Input_Current': 'Input Current',
            'General_Specifications': 'General Specifications',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            'Input_Voltage': 'Input Voltage',
            'Height_Adjustment': 'Height Adjustment',
            'mount_material': 'Construction Material',
            'Plug_Type': 'Plug Type',
            'power_consumption': 'Power Consumption (In Watts)',
            'Output_Voltage': 'Output Voltage',
            'Cable_Management': 'Cable Management',
            'cable_length_raw': 'Cable Length',
            'mount_max_display_size': 'Maximum Display Size',
            'Wallmountable': 'Wallmountable',
            'Fits_UltraWide_Displays': 'Fits UltraWide Displays',
            'Anti-Theft': 'Anti-Theft',
            'Maximum_Height': 'Maximum Height',
            'mount_curved_tv': 'Fits Curved Display',
            'hub_ports_raw': 'Ports',
            'Weight_Capacity_of_Work_Surface': 'Weight Capacity of Work Surface',
            'dock_num_displays': 'Displays Supported',
            'Tools_Included': 'Tools Included',
            'power_adapter': 'Power Source',
            'standards': 'Industry Standards',
            'Compatible_Lock_Slot': 'Compatible Lock Slot',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'Display_Tilt': 'Display Tilt',
            'Swivel_/_Pivot': 'Swivel / Pivot',
            'Number_of_Power_Outlets': 'Number of Power Outlets',
            'Mounting_Surface_Thickness': 'Mounting Surface Thickness',
            'Minimum_Height': 'Minimum Height',
            'Keyboard_Tray_Width': 'Keyboard Tray Width',
            'Keyboard_Tray_Depth': 'Keyboard Tray Depth',
            'Input_Connectors': 'Input Connector(s)',
            'MTBF': 'MTBF',
            'Output_Connectors': 'Output Connectors',
            'Keyboard_Tray_Tilt': 'Keyboard Tray Tilt',
            'Input_Voltage_Frequency': 'Input Voltage Frequency',
            'Display_Rotation': 'Display Rotation',
            'interface_a': 'Connector A',
            'mount_min_display_size': 'Minimum Display Size',
            'Maximum_Arm_Extension': 'Maximum Arm Extension',
            'On-Screen_Display': 'On-Screen Display',
            'power_delivery': 'Power Delivery',
            'Hot-Key_Selection': 'Hot-Key Selection',
            'Regulatory_Derated_Input_Current': 'Regulatory Derated Input Current',
            'Rack-Mountable': 'Rack-Mountable',
            'Nominal_Output_Voltage': 'Nominal Output Voltage',
            'Nominal_Input_Voltage': 'Nominal Input Voltage',
            'Maximum_Total_Current_Draw_per_Phase': 'Maximum Total Current Draw per Phase',
            'Maximum_Total_Current': 'Maximum Total Current',
            'Maximum_Output_Voltage': 'Maximum Output Voltage',
            'interface_b': 'Connector B',
            'Articulating': 'Articulating',
            'Minimum_Profile_from_Wall': 'Minimum Profile from Wall',
            'Minimum_Arm_Extension': 'Minimum Arm Extension',
            'LED_Indicators': 'LED Indicators',
            'Internal_Height': 'Internal Height',
            'Video_Wall': 'Video Wall',
            'mount_vesa_pattern': 'VESA Hole Pattern(s)',
            'Internal_Width': 'Internal Width',
            'io_interface_raw': 'Interface',
            'Frame_Type': 'Frame Type',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'Display_Swivel': 'Display Swivel',
            'connector_style': 'Connector Style',
            'connector_plating': 'Connector Plating',
            'shield_type': 'Cable Shield Material',
            'jacket_type': 'Cable Jacket Material',
        },
    },

    # =========================================================================
    # VIDEO_SPLITTER (13 products)
    # From: "Video Display Connectivity" (13)
    # =========================================================================
    'video_splitter': {
        'tier1': {
            'av_input': 'AV Input',
            'av_output': 'AV Output',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'supported_resolutions': 'Supported Resolutions',
            'power_adapter': 'Power Source',
            'kvm_audio': 'Audio',
        },
        'tier2': {
            'power_adapter': 'Power Source',
            'kvm_audio': 'Audio',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'Output_Voltage': 'Output Voltage',
            'Input_Voltage': 'Input Voltage',
            'supported_resolutions': 'Supported Resolutions',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            'Output_Current': 'Output Current',
            'Wide_Screen_Supported': 'Wide Screen Supported',
            'Plug_Type': 'Plug Type',
            'Input_Current': 'Input Current',
            'power_consumption': 'Power Consumption (In Watts)',
            'hub_ports_raw': 'Ports',
            'audio_ports': 'Audio Specifications',
            'av_input': 'AV Input',
            'av_output': 'AV Output',
            'Cabling': 'Cabling',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'max_distance': 'Max Distance',
            'Daisy-Chain': 'Daisy-Chain',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'interface_a': 'Connector A',
            'standards': 'Industry Standards',
            'Rack-Mountable': 'Rack-Mountable',
            'chipset': 'Chipset ID',
            'LED_Indicators': 'LED Indicators',
            'os_compatibility': 'OS Compatibility',
            'MTBF': 'MTBF',
            'network_speed': 'Maximum Data Transfer Rate',
            'General_Specifications': 'General Specifications',
            'Video_Revision': 'Video Revision',
            'active_passive': 'Active or Passive Adapter',
            'max_resolution': 'Maximum Analog Resolutions',
            'Bandwidth': 'Bandwidth',
            'dock_num_displays': 'Displays Supported',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'Auto_MDIX': 'Auto MDIX',
            'cable_length_raw': 'Cable Length',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'Maximum_Cable_Distance_To_Display': 'Maximum Cable Distance To Display',
            'dock_4k_support': '4K Support',
            'Frequency': 'Frequency',
            'Full_Duplex_Support': 'Full Duplex Support',
            'type_and_rate': 'Type and Rate',
            'Input_Connectors': 'Input Connector(s)',
            'Regulatory_Approvals': 'Regulatory Approvals',
            'UASP_Support': 'UASP Support',
            'nw_cable_type': 'Compatible Networks',
            'Aspect_Ratio': 'Aspect Ratio',
            'conn_type': 'Connector Type(s)',
            'power_delivery': 'Power Delivery',
            'interface_b': 'Connector B',
            'USB_Pass-Through': 'USB Pass-Through',
            'Number_of_Power_Outlets': 'Number of Power Outlets',
            'Maximum_Output_Voltage': 'Maximum Output Voltage',
            'Nominal_Input_Voltage': 'Nominal Input Voltage',
            'Nominal_Output_Voltage': 'Nominal Output Voltage',
            'connectors_desc': 'Connectors',
        },
    },

    # =========================================================================
    # VIDEO_SWITCH (14 products)
    # From: "Video Display Connectivity" (14)
    # =========================================================================
    'video_switch': {
        'tier1': {
            'power_adapter': 'Power Source',
            'kvm_audio': 'Audio',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'Output_Voltage': 'Output Voltage',
            'Input_Voltage': 'Input Voltage',
            'supported_resolutions': 'Supported Resolutions',
        },
        'tier2': {
            'power_adapter': 'Power Source',
            'kvm_audio': 'Audio',
            'max_dvi_resolution': 'Maximum Digital Resolutions',
            'Output_Voltage': 'Output Voltage',
            'Input_Voltage': 'Input Voltage',
            'supported_resolutions': 'Supported Resolutions',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            'Output_Current': 'Output Current',
            'Wide_Screen_Supported': 'Wide Screen Supported',
            'Plug_Type': 'Plug Type',
            'Input_Current': 'Input Current',
            'power_consumption': 'Power Consumption (In Watts)',
            'hub_ports_raw': 'Ports',
            'audio_ports': 'Audio Specifications',
            'av_input': 'AV Input',
            'av_output': 'AV Output',
            'Cabling': 'Cabling',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'max_distance': 'Max Distance',
            'Daisy-Chain': 'Daisy-Chain',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'interface_a': 'Connector A',
            'standards': 'Industry Standards',
            'Rack-Mountable': 'Rack-Mountable',
            'chipset': 'Chipset ID',
            'LED_Indicators': 'LED Indicators',
            'os_compatibility': 'OS Compatibility',
            'MTBF': 'MTBF',
            'network_speed': 'Maximum Data Transfer Rate',
            'General_Specifications': 'General Specifications',
            'Video_Revision': 'Video Revision',
            'active_passive': 'Active or Passive Adapter',
            'max_resolution': 'Maximum Analog Resolutions',
            'Bandwidth': 'Bandwidth',
            'dock_num_displays': 'Displays Supported',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'Auto_MDIX': 'Auto MDIX',
            'cable_length_raw': 'Cable Length',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'Maximum_Cable_Distance_To_Display': 'Maximum Cable Distance To Display',
            'dock_4k_support': '4K Support',
            'Frequency': 'Frequency',
            'Full_Duplex_Support': 'Full Duplex Support',
            'type_and_rate': 'Type and Rate',
            'Input_Connectors': 'Input Connector(s)',
            'Regulatory_Approvals': 'Regulatory Approvals',
            'UASP_Support': 'UASP Support',
            'nw_cable_type': 'Compatible Networks',
            'Aspect_Ratio': 'Aspect Ratio',
            'conn_type': 'Connector Type(s)',
            'power_delivery': 'Power Delivery',
            'interface_b': 'Connector B',
            'USB_Pass-Through': 'USB Pass-Through',
            'Number_of_Power_Outlets': 'Number of Power Outlets',
            'Maximum_Output_Voltage': 'Maximum Output Voltage',
            'Nominal_Input_Voltage': 'Nominal Input Voltage',
            'Nominal_Output_Voltage': 'Nominal Output Voltage',
            'connectors_desc': 'Connectors',
        },
    },

    # =========================================================================
    # MOUNT (13 products) - Generic mounts (non-display)
    # From: "Workstation Accessories" (13)
    # =========================================================================
    'mount': {
        'tier1': {
            'mount_options': 'Mounting Options',
            'material': 'Material',
        },
        'tier2': {
            'mount_options': 'Mounting Options',
            'Screen_Size': 'Screen Size',
            'Aspect_Ratio': 'Aspect Ratio',
            'Orientation': 'Orientation',
            'Output_Current': 'Output Current',
            'Input_Current': 'Input Current',
            'General_Specifications': 'General Specifications',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'humidity': 'Humidity',
            'storage_temp': 'Storage Temperature',
            'Input_Voltage': 'Input Voltage',
            'Height_Adjustment': 'Height Adjustment',
            'mount_material': 'Construction Material',
            'Plug_Type': 'Plug Type',
            'power_consumption': 'Power Consumption (In Watts)',
            'Output_Voltage': 'Output Voltage',
            'Cable_Management': 'Cable Management',
            'cable_length_raw': 'Cable Length',
            'mount_max_display_size': 'Maximum Display Size',
            'Wallmountable': 'Wallmountable',
            'Fits_UltraWide_Displays': 'Fits UltraWide Displays',
            'Anti-Theft': 'Anti-Theft',
            'Maximum_Height': 'Maximum Height',
            'mount_curved_tv': 'Fits Curved Display',
            'hub_ports_raw': 'Ports',
            'Weight_Capacity_of_Work_Surface': 'Weight Capacity of Work Surface',
            'dock_num_displays': 'Displays Supported',
            'Tools_Included': 'Tools Included',
            'power_adapter': 'Power Source',
            'standards': 'Industry Standards',
            'Compatible_Lock_Slot': 'Compatible Lock Slot',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'Display_Tilt': 'Display Tilt',
            'Swivel_/_Pivot': 'Swivel / Pivot',
            'Number_of_Power_Outlets': 'Number of Power Outlets',
            'Mounting_Surface_Thickness': 'Mounting Surface Thickness',
            'Minimum_Height': 'Minimum Height',
            'Keyboard_Tray_Width': 'Keyboard Tray Width',
            'Keyboard_Tray_Depth': 'Keyboard Tray Depth',
            'Input_Connectors': 'Input Connector(s)',
            'MTBF': 'MTBF',
            'Output_Connectors': 'Output Connectors',
            'Keyboard_Tray_Tilt': 'Keyboard Tray Tilt',
            'Input_Voltage_Frequency': 'Input Voltage Frequency',
            'Display_Rotation': 'Display Rotation',
            'interface_a': 'Connector A',
            'mount_min_display_size': 'Minimum Display Size',
            'Maximum_Arm_Extension': 'Maximum Arm Extension',
            'On-Screen_Display': 'On-Screen Display',
            'power_delivery': 'Power Delivery',
            'Hot-Key_Selection': 'Hot-Key Selection',
            'Regulatory_Derated_Input_Current': 'Regulatory Derated Input Current',
            'Rack-Mountable': 'Rack-Mountable',
            'Nominal_Output_Voltage': 'Nominal Output Voltage',
            'Nominal_Input_Voltage': 'Nominal Input Voltage',
            'Maximum_Total_Current_Draw_per_Phase': 'Maximum Total Current Draw per Phase',
            'Maximum_Total_Current': 'Maximum Total Current',
            'Maximum_Output_Voltage': 'Maximum Output Voltage',
            'interface_b': 'Connector B',
            'Articulating': 'Articulating',
            'Minimum_Profile_from_Wall': 'Minimum Profile from Wall',
            'Minimum_Arm_Extension': 'Minimum Arm Extension',
            'LED_Indicators': 'LED Indicators',
            'Internal_Height': 'Internal Height',
            'Video_Wall': 'Video Wall',
            'mount_vesa_pattern': 'VESA Hole Pattern(s)',
            'Internal_Width': 'Internal Width',
            'io_interface_raw': 'Interface',
            'Frame_Type': 'Frame Type',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'Display_Swivel': 'Display Swivel',
            'connector_style': 'Connector Style',
            'connector_plating': 'Connector Plating',
            'shield_type': 'Cable Shield Material',
            'jacket_type': 'Cable Jacket Material',
        },
    },

    # =========================================================================
    # ETHERNET_SWITCH (11 products)
    # From: "Networking" (11)
    # =========================================================================
    'ethernet_switch': {
        'tier1': {
            'nw_cable_type': 'Compatible Networks',
            'network_speed': 'Maximum Data Transfer Rate',
            'poe': 'PoE',
            'standards': 'Industry Standards',
            'power_consumption': 'Power Consumption (In Watts)',
            'max_distance': 'Max Transfer Distance',
        },
        'tier2': {
            'nw_cable_type': 'Compatible Networks',
            'network_speed': 'Maximum Data Transfer Rate',
            'poe': 'PoE',
            'features': 'Features',
            'standards': 'Industry Standards',
            'Compatible_Brand': 'Compatible Brand',
            'power_consumption': 'Power Consumption (In Watts)',
            'fiber_type_raw': 'Fiber Type',
            'Max_Transfer_Distance': 'Max Transfer Distance',
            # Physical dimensions
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'warranty': 'Warranty',
            'color': 'Color',
            'material': 'Material',
            'operating_temp': 'Operating Temperature',
            'storage_temp': 'Storage Temperature',
            'humidity': 'Humidity',
            'MTBF': 'MTBF',
            'Fiber_Connector': 'Fiber Connector',
            'DDM': 'DDM',
            'fiber_duplex_raw': 'Fiber Operating Mode',
            'fiber_wavelength_raw': 'Wavelength',
            'Output_Voltage': 'Output Voltage',
            'Auto_MDIX': 'Auto MDIX',
            'chipset': 'Chipset ID',
            'os_compatibility': 'OS Compatibility',
            'hub_ports_raw': 'Ports',
            'cable_length_raw': 'Cable Length',
            'power_adapter': 'Power Source',
            'System_and_Cable_Requirements': 'System and Cable Requirements',
            'Jumbo_Frame_Support': 'Jumbo Frame Support',
            'Full_Duplex_Support': 'Full Duplex Support',
            'usb_type': 'Bus Type',
            'Input_Voltage': 'Input Voltage',
            'io_interface_raw': 'Interface',
            'jacket_type': 'Cable Jacket Material',
            'Wake_On_Lan': 'Wake On Lan',
            'Impedance': 'Impedance',
            'wire_gauge': 'Wire Gauge',
            'Local_Unit_Connectors': 'Local Unit Connectors',
            'WDM': 'WDM',
            'Promiscuous_Mode': 'Promiscuous Mode',
            'Plug_Type': 'Plug Type',
            'PXE': 'PXE',
            'Output_Current': 'Output Current',
            'Input_Current': 'Input Current',
            'form_factor': 'Form Factor',
            'USB_Ports_Supported': 'USB Ports Supported',
            'poe': 'PoE',
            'General_Specifications': 'General Specifications',
            'Maximum_Cable_Length': 'Maximum Cable Length',
            'Supported_Protocols': 'Supported Protocols',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'type_and_rate': 'Type and Rate',
            'max_distance': 'Max Distance',
            'Flow_Control': 'Flow Control',
            'external_ports': 'External Ports',
            'Switching_Architecture': 'Switching Architecture',
            'LED_Indicators': 'LED Indicators',
            'Port_Style': 'Port Style',
            'Antenna_Configuration': 'Antenna Configuration',
            'Buffer_Size': 'Buffer Size',
            'conn_type': 'Connector Type(s)',
            'Surge_Protection': 'Surge Protection',
            'USB_Pass-Through': 'USB Pass-Through',
            'Frequency': 'Frequency',
            'Frequency_Range': 'Frequency Range',
            'interface_a': 'Connector A',
            'Remote_Management_Ability': 'Remote Management Ability',
            'Bandwidth': 'Bandwidth',
            'Auto-Negotiation': 'Auto-Negotiation',
            'interface_b': 'Connector B',
            'Cabling': 'Cabling',
            'Internal_Ports': 'Internal Ports',
            'Rack-Mountable': 'Rack-Mountable',
            'Anti-Theft': 'Anti-Theft',
            'Cable_Management': 'Cable Management',
            'dock_fast_charge': 'Fast-Charge Port(s)',
            'power_delivery': 'Power Delivery',
            'Fans': 'Fan(s)',
            'Front_Door_Construction': 'Front Door Construction',
            'Front_Door_Features': 'Front Door Features',
            'Height_Adjustment': 'Height Adjustment',
            'Wallmountable': 'Wallmountable',
            'Tools_Included': 'Tools Included',
            'Regulatory_Approvals': 'Regulatory Approvals',
            'Compatible_Lock_Slot': 'Compatible Lock Slot',
            'UASP_Support': 'UASP Support',
            'USB-C_Device_Ports': 'USB-C Device Port(s)',
            'USB-C_Host_Connection': 'USB-C Host Connection',
            'mount_options': 'Mounting Options',
        },
    },

    # =========================================================================
    # OTHER (1,162 products) - Miscellaneous products
    # From: "No Category" (917), "Workstation Accessories" (126),
    #       "Legacy Category" (46), "Video Display Connectivity" (43),
    #       "Data Storage" (30)
    # =========================================================================
    'other': {
        'tier1': {
            'Input_Voltage': 'Input Voltage',
            'Output_Current': 'Output Current',
            'Output_Voltage': 'Output Voltage',
            'Plug_Type': 'Plug Type',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'product_length': 'Product Length',
        },
        'tier2': {
            'Input_Voltage': 'Input Voltage',
            'Output_Current': 'Output Current',
            'Output_Voltage': 'Output Voltage',
            'Plug_Type': 'Plug Type',
            'Center_Tip_Polarity': 'Center Tip Polarity',
            'product_length': 'Product Length',
            'product_width': 'Product Width',
            'product_height': 'Product Height',
            'product_weight': 'Weight of Product',
            'package_length': 'Package Length',
            'package_width': 'Package Width',
            'package_height': 'Package Height',
            'package_weight': 'Shipping (Package) Weight',
            'Package_Quantity': 'Package Quantity',
            'Included_in_Package': 'Included in Package',
            'color': 'Color',
            'warranty': 'Warranty',
            'material': 'Material',
            'Input_Current': 'Input Current',
            'cable_length_raw': 'Cable Length',
            'jacket_type': 'Cable Jacket Material',
            'fire_rating': 'Fire Rating',
        },
    },
}


# =============================================================================
# TIER 3 FIELDS — Physical dimensions, packaging, and warranty
# These are excluded from Tier 2 (differences/specs) and only included
# in Tier 3 (single-product detail views).
# =============================================================================

TIER3_FIELDS = {
    'product_length',
    'product_width',
    'product_height',
    'product_weight',
    'package_length',
    'package_width',
    'package_height',
    'package_weight',
    'Package_Quantity',
    'Included_in_Package',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tier1_columns(category: str) -> List[str]:
    """Get Tier 1 display columns for a category (quick overview).

    Falls back to 'other' if category not found.
    """
    config = CATEGORY_COLUMNS.get(category, CATEGORY_COLUMNS.get('other', {}))
    tier = config.get('tier1', {})
    return list(tier.keys()) if isinstance(tier, dict) else tier


def get_tier1_config(category: str) -> dict:
    """Get Tier 1 config dict {field_name: display_label} for a category.

    Use this when you need both field names and their display labels.
    Falls back to 'other' if category not found.
    """
    config = CATEGORY_COLUMNS.get(category, CATEGORY_COLUMNS.get('other', {}))
    tier = config.get('tier1', {})
    return dict(tier) if isinstance(tier, dict) else {}


def get_tier2_columns(category: str) -> List[str]:
    """Get Tier 2 display columns for a category (specs without physical/packaging).

    Tier 2 is used for highlighting differences between products.
    Excludes TIER3_FIELDS (physical dimensions, packaging, warranty).
    Falls back to 'other' if category not found.
    """
    config = CATEGORY_COLUMNS.get(category, CATEGORY_COLUMNS.get('other', {}))
    tier = config.get('tier2', {})
    all_cols = list(tier.keys()) if isinstance(tier, dict) else tier
    return [col for col in all_cols if col not in TIER3_FIELDS]


def get_tier3_columns(category: str) -> List[str]:
    """Get Tier 3 display columns for a category (all fields including physical).

    Tier 3 is used for single-product detail views.
    Returns the full column set (Tier 2 + physical/packaging/warranty).
    Falls back to 'other' if category not found.
    """
    config = CATEGORY_COLUMNS.get(category, CATEGORY_COLUMNS.get('other', {}))
    tier = config.get('tier2', {})
    return list(tier.keys()) if isinstance(tier, dict) else tier


def get_field_label(field: str, category: str = '') -> str:
    """Get human-readable label for a metadata field."""
    if category:
        config = CATEGORY_COLUMNS.get(category, {})
        for tier_key in ['tier1', 'tier2']:
            tier_data = config.get(tier_key, {})
            if isinstance(tier_data, dict) and field in tier_data:
                return tier_data[field]
    for cat_config in CATEGORY_COLUMNS.values():
        for tier_key in ['tier1', 'tier2']:
            tier_data = cat_config.get(tier_key, {})
            if isinstance(tier_data, dict) and field in tier_data:
                return tier_data[field]
    return field.replace('_', ' ').strip().title()


def get_columns_for_product(product_metadata: dict, tier: int = 1) -> dict:
    """
    Extract the relevant columns for a product at the specified tier.

    Args:
        product_metadata: The product's metadata dict
        tier: 1 for search results, 2 for specs/differences, 3 for full detail

    Returns:
        Dict of {column_name: value} for columns that have non-empty values
    """
    category = product_metadata.get('category', 'other')
    if tier == 1:
        columns = get_tier1_columns(category)
    elif tier == 3:
        columns = get_tier3_columns(category)
    else:
        columns = get_tier2_columns(category)

    result = {}
    for col in columns:
        value = product_metadata.get(col)
        if value is not None and str(value).strip().lower() not in ('', 'nan', 'none'):
            result[col] = value
    return result


def get_all_categories() -> List[str]:
    """Get list of all configured categories."""
    return list(CATEGORY_COLUMNS.keys())


def validate_config(products) -> List[str]:
    """Validate category_columns config against actual product metadata.

    Checks that tier1/tier2 field keys exist in at least one product's metadata
    for each category. Returns a list of warning strings for mismatches.

    Call at startup after loading products to catch config issues early.
    """
    warnings = []
    # Group products by category
    by_category = {}
    for p in products:
        cat = p.metadata.get('category', '')
        if cat:
            by_category.setdefault(cat, []).append(p)

    for category, config in CATEGORY_COLUMNS.items():
        cat_products = by_category.get(category, [])
        if not cat_products:
            continue  # No products in this category — skip

        # Collect all metadata keys across products in this category
        all_keys = set()
        for p in cat_products:
            all_keys.update(p.metadata.keys())

        # Check tier1 and tier2 fields
        for tier_name in ('tier1', 'tier2'):
            tier = config.get(tier_name, {})
            if not isinstance(tier, dict):
                continue
            for field in tier:
                if field not in all_keys:
                    warnings.append(
                        f"[{category}] {tier_name} field '{field}' not found in any product metadata"
                    )

    return warnings
