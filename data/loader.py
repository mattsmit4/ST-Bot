"""
Excel Loader for StarTech.com Product Data.

Loads ProductAttributeValues_Cleaned_Exported.xlsx with 5,162 products and 267 columns.
Maps to ST-Bot's Product model.

Architecture: Load ALL columns, normalize display-critical fields.
- All 267 columns are stored in metadata (for answering any question)
- Display-critical fields get normalized aliases (for clean customer display)
- Empty/NaN values are not stored (keeps memory efficient)
- COLUMN_ALIASES is the SINGLE SOURCE OF TRUTH for column name mapping.
"""

import pandas as pd
from typing import List
from core.models import Product
from data.derived import compute_derived_fields


# =============================================================================
# COLUMN NORMALIZATION MAPPINGS
# =============================================================================

COLUMN_ALIASES = {
    # Core identification
    'ProductNumber': 'sku',
    'ItemCategory': 'excel_category',
    'ItemSubCategory': 'sub_category',
    'Description_of_Product': 'description',

    # Length fields
    'Cable_Length': 'cable_length_raw',

    # Connector fields
    'Connector_A': 'interface_a',
    'Connector_B': 'interface_b',
    'Host_Connectors': 'host_connector',
    'Connector_Types': 'conn_type',
    'Connectors': 'connectors_desc',
    'External_Ports': 'external_ports',

    # Network cable fields
    'Cable_Rating': 'network_rating_raw',
    'Maximum_Data_Transfer_Rate': 'network_speed',
    'Compatible_Networks': 'nw_cable_type',

    # Hub-specific fields
    'Ports': 'hub_ports_raw',
    'Total_Ports': 'total_ports',
    'Bus_Type': 'usb_type',
    'Power_Source': 'power_adapter',
    'Power_Delivery': 'power_delivery',

    # KVM-specific fields
    'KVM_Ports': 'kvm_ports_raw',
    'PC_Interface': 'kvm_interface',
    'Audio': 'kvm_audio',
    'PC_Video_Type': 'kvm_video_type',

    # Mount-specific fields
    'Maximum_Display_Size': 'mount_max_display_size',
    'Minimum_Display_Size': 'mount_min_display_size',
    'VESA_Hole_Patterns': 'mount_vesa_pattern',
    'Number_of_Monitors_Supported': 'mount_num_displays',
    'Mounting_Options': 'mount_options',
    'Construction_Material': 'mount_material',
    'Fits_Curved_Display': 'mount_curved_tv',

    # Fiber cable-specific fields
    'Fiber_Type': 'fiber_type_raw',
    'Fiber_Operating_Mode': 'fiber_duplex_raw',
    'Wavelength': 'fiber_wavelength_raw',

    # Storage enclosure-specific fields
    'Drive_Connectors': 'drive_connector_raw',
    'Drive_Size': 'drive_size_raw',
    'Number_of_Drives': 'num_drives_raw',
    'Interface': 'io_interface_raw',
    'Compatible_Drive_Types': 'drive_compatibility_raw',

    # Extended specs
    'Wire_Gauge': 'wire_gauge',
    'Connector_Plating': 'connector_plating',
    'Maximum_Analog_Resolutions': 'max_resolution',
    'Maximum_Digital_Resolutions': 'max_dvi_resolution',
    'Supported_Resolutions': 'supported_resolutions',
    'Cable_Jacket_Material': 'jacket_type',
    'Connector_Style': 'connector_style',
    'Color': 'color',
    'Cable_Shield_Material': 'shield_type',
    'Conductor_Type': 'conductor_type',
    'Fire_Rating': 'fire_rating',
    'Warranty': 'warranty',
    'Industry_Standards': 'standards',
    'Material': 'material',

    # Physical dimensions
    'Product_Height': 'product_height',
    'Product_Length': 'product_length',
    'Product_Width': 'product_width',
    'Weight_of_Product': 'product_weight',
    'Package_Height': 'package_height',
    'Package_Length': 'package_length',
    'Package_Width': 'package_width',
    'Shipping_Package_Weight': 'package_weight',

    # Environmental specs
    'Operating_Temperature': 'operating_temp',
    'Storage_Temperature': 'storage_temp',
    'Humidity': 'humidity',

    # Dock/adapter-specific fields
    '4K_Support': 'dock_4k_support',
    'Fast-Charge_Ports': 'dock_fast_charge',
    'Displays_Supported': 'dock_num_displays',
    'Active_or_Passive_Adapter': 'active_passive',

    # Network features
    'PoE': 'poe',
    'Audio_Specifications': 'audio_ports',

    # Additional useful fields
    'Form_Factor': 'form_factor',
    'Type_and_Rate': 'type_and_rate',
    'Max_Distance': 'max_distance',

    # Rack-specific fields
    'U_Height': 'u_height',
    'Rack_Type': 'rack_type',

    # Card-specific fields
    'Chipset_ID': 'chipset',

    # Power fields
    'Power_Consumption_In_Watts': 'power_consumption',

    # AV/Video splitter fields
    'AV_Input': 'av_input',
    'AV_Output': 'av_output',
}


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_startech_products(excel_path: str) -> List[Product]:
    """
    Load products from StarTech.com Excel file.

    Args:
        excel_path: Path to ProductAttributeValues_Cleaned_Exported.xlsx

    Returns:
        List of Product objects with complete metadata
    """
    print(f"Loading products from: {excel_path}")

    df = pd.read_excel(excel_path)
    print(f"Total rows in Excel: {len(df)}")
    print(f"Total columns in Excel: {len(df.columns)}")

    products = []
    skipped = 0
    errors = []

    for idx, row in df.iterrows():
        try:
            sku = row.get('ProductNumber')
            if pd.isna(sku):
                skipped += 1
                if len(errors) < 10:
                    errors.append(f"Row {idx}: No SKU")
                continue

            sku = str(sku).strip()

            # STEP 1: Load ALL columns into metadata
            metadata = {}
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    if hasattr(val, 'item'):
                        val = val.item()
                    key = COLUMN_ALIASES.get(col, col)
                    metadata[key] = val

            # STEP 2: Compute derived/normalized fields
            compute_derived_fields(row, metadata)

            # STEP 3: Build product name
            name = sku
            metadata['name'] = name
            metadata['sku'] = sku

            # STEP 4: Build content string for search
            content_parts = [name]
            category = metadata.get('category')
            if category:
                content_parts.append(f"Category: {category}")
            length_display = metadata.get('length_display')
            if length_display:
                content_parts.append(f"Length: {length_display}")
            connectors = metadata.get('connectors')
            if connectors and len(connectors) >= 2:
                conn_str = f"{connectors[0]} to {connectors[1]}"
                content_parts.append(f"Connectors: {conn_str}")
            features = metadata.get('features', [])
            if features:
                content_parts.append(f"Features: {', '.join(features)}")

            content = " | ".join(content_parts)

            # STEP 5: Create Product object
            product = Product(
                product_number=sku,
                content=content,
                metadata=metadata,
                score=1.0
            )
            products.append(product)

        except Exception as e:
            skipped += 1
            if len(errors) < 10:
                errors.append(f"Row {idx}: {type(e).__name__}: {str(e)}")
            continue

    print(f"Successfully loaded {len(products)} products")
    if skipped > 0:
        print(f"Skipped {skipped} products due to errors")
        if errors:
            for err in errors[:5]:
                print(f"  - {err}")

    return products


def get_product_statistics(products: List[Product]) -> dict:
    """Get statistics about loaded products."""
    stats = {
        'total': len(products),
        'by_category': {},
        'with_length': 0,
        'with_connectors': 0,
        'with_features': 0,
        'avg_metadata_fields': 0,
    }

    total_fields = 0
    for product in products:
        category = product.metadata.get('category', 'other')
        stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
        if product.metadata.get('length_ft'):
            stats['with_length'] += 1
        if product.metadata.get('connectors'):
            stats['with_connectors'] += 1
        if product.metadata.get('features'):
            stats['with_features'] += 1
        total_fields += len(product.metadata)

    if products:
        stats['avg_metadata_fields'] = round(total_fields / len(products), 1)

    return stats
