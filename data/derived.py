"""
Derived field computation for StarTech.com product data.

Parsing helpers and derived field logic extracted from loader.py.
These functions operate on raw pandas rows and metadata dicts —
they do NOT depend on the Product model.
"""

import pandas as pd
import re
from typing import List, Optional, Any


# =============================================================================
# PARSING HELPERS
# =============================================================================

def _normalize_usb_version(raw_str) -> Optional[str]:
    """Normalize a raw string to a standard USB/Thunderbolt version.
    Returns None for connector-only values like USB-C, USB-A."""
    if not raw_str:
        return None
    lower = str(raw_str).strip().lower()
    if not lower or lower == 'nan':
        return None
    # Compound protocols first
    if 'thunderbolt' in lower and 'usb4' in lower:
        return 'Thunderbolt / USB4'
    if 'usb 3.2 gen 2' in lower or '10 gbit' in lower or '10gbps' in lower:
        return 'USB 3.2 Gen 2 (10Gbps)'
    if 'usb 3.2 gen 1' in lower or 'usb 3.1' in lower or 'usb 3.0' in lower or '5 gbit' in lower or '5gbps' in lower:
        return 'USB 3.0 (5Gbps)'
    if 'usb 2.0' in lower:
        return 'USB 2.0'
    if 'usb4' in lower or 'usb 4' in lower:
        return 'USB4'
    if 'thunderbolt 4' in lower:
        return 'Thunderbolt 4'
    if 'thunderbolt 3' in lower:
        return 'Thunderbolt 3'
    return None


def parse_cable_length(length_str) -> tuple:
    """
    Parse cable length from various formats.
    Returns tuple: (feet, meters)

    StarTech Excel stores lengths in millimeters as bare numbers.
    """
    if pd.isna(length_str):
        return None, None

    try:
        length_str = str(length_str).lower().strip()
        length_str = length_str.replace('approx', '').replace('approximately', '').strip()

        # Millimeters explicitly
        if 'mm' in length_str:
            match = re.search(r'([\d.]+)\s*mm', length_str)
            if match:
                mm = float(match.group(1))
                feet = round(mm / 304.8, 1)
                meters = round(mm / 1000.0, 1)
                return feet, meters

        # Meters
        if 'm' in length_str:
            match = re.search(r'([\d.]+)\s*m', length_str)
            if match:
                meters = float(match.group(1))
                feet = round(meters * 3.28084, 1)
                return feet, meters

        # Inches
        if 'in' in length_str or '"' in length_str:
            match = re.search(r'([\d.]+)\s*(?:in|")', length_str)
            if match:
                inches = float(match.group(1))
                feet = round(inches / 12.0, 1)
                meters = round(inches * 0.0254, 1)
                return feet, meters

        # Feet explicitly
        if 'ft' in length_str or 'foot' in length_str or 'feet' in length_str or "'" in length_str:
            match = re.search(r'([\d.]+)', length_str)
            if match:
                feet = float(match.group(1))
                meters = round(feet / 3.28084, 1)
                return feet, meters

        # Bare number — assume millimeters (StarTech Excel format)
        match = re.search(r'([\d.]+)', length_str)
        if match:
            mm = float(match.group(1))
            if mm > 10:  # Likely in mm
                feet = round(mm / 304.8, 1)
                meters = round(mm / 1000.0, 1)
                return feet, meters
            else:  # Small number, might already be in feet
                feet = mm
                meters = round(feet / 3.28084, 1)
                return feet, meters

        return None, None

    except Exception:
        return None, None


def extract_features(row: pd.Series) -> List[str]:
    """Extract searchable/filterable features from product data."""
    features = []

    # Resolution support
    max_res = row.get('Maximum_Analog_Resolutions') or row.get('Maximum_Digital_Resolutions')
    sup_res = row.get('Supported_Resolutions')
    res_str = ''
    if pd.notna(max_res):
        res_str += str(max_res).upper()
    if pd.notna(sup_res):
        res_str += ' ' + str(sup_res).upper()

    if res_str:
        if '3840' in res_str or '4K' in res_str or '2160' in res_str or 'ULTRA HD' in res_str:
            features.append('4K')
        elif '7680' in res_str or '8K' in res_str or '4320' in res_str:
            features.append('8K')
        elif '2560' in res_str or '1440' in res_str:
            features.append('1440p')
        elif '1920' in res_str or '1080' in res_str:
            features.append('1080p')

        # Extract refresh rate
        hz_patterns = [
            r'@\s*(\d+)\s*Hz',
            r'(\d+)\s*Hz',
            r'p(\d+)\b',
        ]
        max_hz = 0
        for pattern in hz_patterns:
            matches = re.findall(pattern, res_str, re.IGNORECASE)
            for match in matches:
                hz = int(match)
                if hz > max_hz and hz <= 240:
                    max_hz = hz

        if max_hz >= 120:
            features.append(f'{max_hz}Hz')
        elif max_hz >= 60:
            features.append('60Hz')
        elif max_hz > 0 and max_hz < 60:
            features.append('30Hz')

    # Power Delivery
    if row.get('Power_Delivery') == 'Yes' or row.get('Fast-Charge_Ports') == 'Yes':
        features.append('Power Delivery')

    # 4K Support (for docks)
    if row.get('4K_Support') == 'Yes':
        features.append('4K')

    # HDR
    if pd.notna(sup_res) and 'HDR' in str(sup_res).upper():
        features.append('HDR')

    # PoE — check column first, then fall back to description
    if row.get('PoE') == 'Yes':
        features.append('PoE')
    elif 'PoE' not in features:
        desc = str(row.get('Description_of_Product', ''))
        if re.search(r'\bPoE\+?\b', desc):
            features.append('PoE')

    # Network Speed
    network_speed = row.get('Maximum_Data_Transfer_Rate')
    if pd.notna(network_speed):
        speed_str = str(network_speed)
        if 'Gigabit' in speed_str or '1000' in speed_str or '1Gbps' in speed_str:
            features.append('Gigabit')
        elif '10G' in speed_str or '10 G' in speed_str:
            features.append('10 Gigabit')

    # Thunderbolt
    if pd.notna(row.get('Host_Connectors')):
        host = str(row.get('Host_Connectors')).lower()
        if 'thunderbolt' in host:
            features.append('Thunderbolt')

    # USB-C
    if pd.notna(row.get('Bus_Type')):
        usb = str(row.get('Bus_Type')).lower()
        if 'usb-c' in usb or 'usb c' in usb or 'type-c' in usb or 'type c' in usb:
            features.append('USB-C')

    # Active cables
    if pd.notna(row.get('Active_or_Passive_Adapter')):
        if str(row.get('Active_or_Passive_Adapter')).lower() == 'active':
            features.append('Active')

    # Shielded
    if pd.notna(row.get('Cable_Shield_Material')):
        shield = str(row.get('Cable_Shield_Material'))
        if shield and shield != 'nan' and shield != 'Unshielded':
            features.append('Shielded')

    # DP Alt Mode
    conn_a_str = str(row.get('Connector_A', '')).lower()
    conn_b_str = str(row.get('Connector_B', '')).lower()
    desc_str = str(row.get('Description_of_Product', '')).lower()
    if 'alt mode' in conn_a_str or 'alt mode' in conn_b_str or 'alt mode' in desc_str:
        features.append('DP Alt Mode')

    # HDCP
    if pd.notna(row.get('Industry_Standards')):
        standards = str(row.get('Industry_Standards')).upper()
        if 'HDCP' in standards:
            features.append('HDCP')

    # Audio
    if row.get('Audio') == 'Yes':
        features.append('Audio')
    if pd.notna(row.get('Audio_Specifications')):
        audio_ports = str(row.get('Audio_Specifications'))
        if audio_ports and audio_ports != 'nan' and audio_ports != '0':
            features.append('Audio')

    return list(set(features))


def extract_connectors(row: pd.Series) -> Optional[list]:
    """
    Extract connector information from Connector_A and Connector_B.
    Falls back to Host_Connectors for multiport adapters and docks.
    """
    interface_a = row.get('Connector_A')
    interface_b = row.get('Connector_B')
    host_connector = row.get('Host_Connectors')

    if pd.isna(interface_a) and pd.isna(interface_b):
        if pd.notna(host_connector):
            host_str = str(host_connector).strip()
            if host_str and host_str.lower() != 'nan':
                return [host_str]
        return None

    connectors = []
    if pd.notna(interface_a):
        source = str(interface_a).strip()
        if source and source != 'nan':
            connectors.append(source)

    if pd.notna(interface_b):
        target = str(interface_b).strip()
        if target and target != 'nan':
            connectors.append(target)

    if len(connectors) == 0:
        return None
    elif len(connectors) == 1:
        connectors.append(connectors[0])

    return connectors


def extract_network_rating(row: pd.Series) -> Optional[dict]:
    """Extract network cable rating (Cat5e, Cat6, Cat6a, etc.)."""
    rating_raw = row.get('Cable_Rating')
    if pd.isna(rating_raw):
        return None

    rating_str = str(rating_raw).strip()
    if not rating_str or rating_str == 'nan':
        return None

    result = {'rating': None, 'rating_full': rating_str, 'max_speed': None}
    rating_upper = rating_str.upper()

    if 'CAT6A' in rating_upper:
        result['rating'] = 'Cat6a'
        result['max_speed'] = '10 Gigabit'
    elif 'CAT6' in rating_upper:
        result['rating'] = 'Cat6'
        result['max_speed'] = 'Gigabit'
    elif 'CAT5E' in rating_upper:
        result['rating'] = 'Cat5e'
        result['max_speed'] = 'Gigabit'
    elif 'CAT5' in rating_upper:
        result['rating'] = 'Cat5'
        result['max_speed'] = '100 Mbps'
    elif 'CAT7' in rating_upper:
        result['rating'] = 'Cat7'
        result['max_speed'] = '10 Gigabit'

    return result if result['rating'] else None


def determine_category(row: pd.Series) -> str:
    """Delegates to core.category_config for single source of truth."""
    from core.category_config import determine_category as _determine_category
    return _determine_category(row)


# =============================================================================
# DERIVED FIELD COMPUTATION
# =============================================================================

def compute_derived_fields(row: pd.Series, metadata: dict) -> None:
    """Compute derived/normalized fields from raw data. Modifies metadata in place."""

    # --- Length fields ---
    length_raw = row.get('Cable_Length')
    length_ft, length_m = parse_cable_length(length_raw)

    if length_ft is not None:
        metadata['length'] = length_ft
        metadata['length_ft'] = length_ft
        metadata['length_m'] = length_m
        metadata['length_unit'] = 'ft'

        if length_m and length_m >= 1:
            metadata['length_display'] = f"{length_ft} ft [{length_m} m]"
        else:
            metadata['length_display'] = f"{length_ft} ft"

    # --- Category ---
    metadata['category'] = determine_category(row)

    # --- Connectors ---
    connectors = extract_connectors(row)
    if connectors:
        metadata['connectors'] = connectors

    # --- Features ---
    features = extract_features(row)
    metadata['features'] = features

    # --- Max refresh rate ---
    max_res = row.get('Maximum_Analog_Resolutions') or row.get('Maximum_Digital_Resolutions') or ''
    sup_res = row.get('Supported_Resolutions') or ''
    if pd.notna(max_res) or pd.notna(sup_res):
        res_text = f"{max_res or ''} {sup_res or ''}".upper()
        hz_patterns = [r'@\s*(\d+)\s*Hz', r'(\d+)\s*Hz', r'p(\d+)\b']
        max_hz = 0
        for pattern in hz_patterns:
            matches = re.findall(pattern, res_text, re.IGNORECASE)
            for match in matches:
                hz = int(match)
                if hz > max_hz and hz <= 240:
                    max_hz = hz
        if max_hz > 0:
            metadata['max_refresh_rate'] = max_hz

    # --- Network rating ---
    network_info = extract_network_rating(row)
    if network_info:
        metadata['network_rating'] = network_info['rating']
        metadata['network_rating_full'] = network_info['rating_full']
        metadata['network_max_speed'] = network_info['max_speed']

    # --- Hub-specific derived fields ---
    num_ports = row.get('Ports')
    if pd.notna(num_ports):
        try:
            metadata['hub_ports'] = int(float(num_ports))
        except (ValueError, TypeError):
            pass

    usb_type = row.get('Bus_Type')
    if pd.notna(usb_type):
        usb_str = str(usb_type).strip()
        if usb_str and usb_str != 'nan':
            metadata['hub_usb_type'] = usb_str

    # Derive hub_usb_version from Interface (reliable version/speed) first,
    # then fall back to Bus_Type. Never stores connector types (USB-C, USB-A).
    for source in (metadata.get('io_interface_raw'), metadata.get('hub_usb_type')):
        version = _normalize_usb_version(source)
        if version:
            metadata['hub_usb_version'] = version
            break

    power_adapter = row.get('Power_Source')
    if pd.notna(power_adapter):
        power_str = str(power_adapter).strip()
        if power_str and power_str != 'nan':
            metadata['hub_power_type'] = power_str
            power_lower = power_str.lower()
            if 'ac adapter' in power_lower or 'included' in power_lower:
                metadata['hub_powered'] = True
            elif 'usb-powered' in power_lower or 'bus-powered' in power_lower or 'bus powered' in power_lower:
                metadata['hub_powered'] = False

    power_delivery = row.get('Power_Delivery')
    if pd.notna(power_delivery):
        pd_str = str(power_delivery).strip()
        if pd_str and pd_str != 'nan' and pd_str.lower() != 'no':
            metadata['hub_power_delivery'] = pd_str

    # --- KVM-specific derived fields ---
    kvm_ports = row.get('KVM_Ports')
    if pd.notna(kvm_ports):
        try:
            metadata['kvm_ports'] = int(float(kvm_ports))
        except (ValueError, TypeError):
            pass

    kvm_audio = row.get('Audio')
    if pd.notna(kvm_audio):
        metadata['kvm_audio'] = str(kvm_audio).strip() == 'Yes'

    # --- Mount-specific derived fields ---
    max_display = row.get('Maximum_Display_Size')
    min_display = row.get('Minimum_Display_Size')
    if pd.notna(max_display):
        try:
            max_str = str(max_display).strip()
            max_match = re.search(r'([\d.]+)', max_str)
            if max_match:
                max_size = float(max_match.group(1))
                metadata['mount_max_display'] = max_size

                if pd.notna(min_display):
                    min_str = str(min_display).strip()
                    min_match = re.search(r'([\d.]+)', min_str)
                    if min_match:
                        min_size = float(min_match.group(1))
                        metadata['mount_min_display'] = min_size
                        metadata['mount_display_range'] = f'{int(min_size)}-{int(max_size)}"'
                    else:
                        metadata['mount_display_range'] = f'Up to {int(max_size)}"'
                else:
                    metadata['mount_display_range'] = f'Up to {int(max_size)}"'
        except (ValueError, TypeError):
            pass

    # VESA pattern
    vesa = row.get('VESA_Hole_Patterns')
    if pd.notna(vesa):
        vesa_str = str(vesa).strip()
        if vesa_str and vesa_str != 'nan':
            vesa_patterns = set()
            for pattern in re.findall(r'(\d+x\d+)', vesa_str):
                vesa_patterns.add(pattern)
            if vesa_patterns:
                sorted_patterns = sorted(vesa_patterns, key=lambda x: int(x.split('x')[0]))
                metadata['mount_vesa'] = ', '.join(sorted_patterns)

    # Number of displays
    num_displays = row.get('Number_of_Monitors_Supported')
    if pd.notna(num_displays):
        try:
            metadata['mount_num_displays'] = int(float(num_displays))
        except (ValueError, TypeError):
            pass

    # Mount type
    mount_options = row.get('Mounting_Options')
    if pd.notna(mount_options):
        mount_str = str(mount_options).strip()
        if mount_str and mount_str != 'nan':
            mount_str = mount_str.replace('&amp;', '&')
            metadata['mount_type'] = mount_str

    # Curved TV support
    curved_tv = row.get('Fits_Curved_Display')
    if pd.notna(curved_tv) and str(curved_tv).strip() == 'Yes':
        metadata['mount_curved_support'] = True

    # Material
    material = row.get('Construction_Material')
    if pd.notna(material):
        mat_str = str(material).strip()
        if mat_str and mat_str != 'nan':
            metadata['mount_material'] = mat_str

    # --- Wire gauge normalization ---
    wire_gauge = row.get('Wire_Gauge')
    if pd.notna(wire_gauge):
        gauge_str = str(wire_gauge).strip()
        if gauge_str and gauge_str != 'nan':
            if 'AWG' not in gauge_str.upper():
                gauge_str = f"{gauge_str} AWG"
            metadata['wire_gauge'] = gauge_str

    # --- Fiber cable-specific derived fields ---
    fiber_type = row.get('Fiber_Type')
    if pd.notna(fiber_type):
        ft_str = str(fiber_type).strip()
        if ft_str and ft_str != 'nan':
            if 'multi' in ft_str.lower():
                metadata['fiber_type'] = 'Multimode'
            elif 'single' in ft_str.lower():
                metadata['fiber_type'] = 'Single-mode'
            else:
                metadata['fiber_type'] = ft_str

    wavelength = row.get('Wavelength')
    if pd.notna(wavelength):
        wl_str = str(wavelength).strip()
        if wl_str and wl_str != 'nan':
            wl_parts = wl_str.split(',')
            if wl_parts:
                wavelengths = []
                for part in wl_parts:
                    match = re.search(r'(\d+)\s*nm', part)
                    if match:
                        wavelengths.append(int(match.group(1)))
                if wavelengths:
                    primary_wl = min(wavelengths)
                    metadata['fiber_wavelength'] = f"{primary_wl}nm"

    # Fiber connector type
    interface_a = row.get('Connector_A')
    if pd.notna(interface_a):
        ia_str = str(interface_a).strip().lower()
        if 'fiber' in ia_str or 'mpo' in ia_str or 'mtp' in ia_str:
            if 'mpo' in ia_str or 'mtp' in ia_str:
                metadata['fiber_connector'] = 'MPO/MTP'
            elif 'lc' in ia_str:
                metadata['fiber_connector'] = 'LC'
            elif 'sc' in ia_str:
                metadata['fiber_connector'] = 'SC'
            elif 'st' in ia_str:
                metadata['fiber_connector'] = 'ST'

            if 'duplex' in ia_str:
                metadata['fiber_duplex'] = 'Duplex'
            elif 'simplex' in ia_str:
                metadata['fiber_duplex'] = 'Simplex'

    # --- Storage enclosure-specific derived fields ---
    drive_size = row.get('Drive_Size')
    if pd.notna(drive_size):
        ds_str = str(drive_size).strip()
        if ds_str and ds_str != 'nan':
            ds_str = ds_str.replace('&amp;', '&')
            if 'm.2' in ds_str.lower() or 'nvme' in ds_str.lower():
                if 'nvme' in ds_str.lower():
                    metadata['drive_size'] = 'M.2 NVMe'
                elif 'sata' in ds_str.lower():
                    metadata['drive_size'] = 'M.2 SATA'
                else:
                    metadata['drive_size'] = 'M.2'
            elif '2.5' in ds_str:
                if '3.5' in ds_str:
                    metadata['drive_size'] = '2.5"/3.5"'
                else:
                    metadata['drive_size'] = '2.5"'
            elif '3.5' in ds_str:
                metadata['drive_size'] = '3.5"'
            elif 'msata' in ds_str.lower():
                metadata['drive_size'] = 'mSATA'
            else:
                metadata['drive_size'] = ds_str

    num_drives = row.get('Number_of_Drives')
    if pd.notna(num_drives):
        try:
            metadata['num_drives'] = int(float(num_drives))
        except (ValueError, TypeError):
            pass

    io_interface = row.get('Interface')
    if pd.notna(io_interface):
        io_str = str(io_interface).strip()
        if io_str and io_str != 'nan':
            io_str = io_str.replace('&amp;', '&')
            io_lower = io_str.lower()
            if 'thunderbolt 3' in io_lower or 'thunderbolt3' in io_lower:
                metadata['storage_interface'] = 'Thunderbolt 3'
            elif 'thunderbolt 4' in io_lower:
                metadata['storage_interface'] = 'Thunderbolt 4'
            elif 'usb 3.2 gen 2' in io_lower or '10gbps' in io_lower or '10 gbit' in io_lower:
                metadata['storage_interface'] = 'USB 3.2 Gen 2 (10Gbps)'
            elif 'usb 3.2 gen 1' in io_lower or 'usb 3.0' in io_lower or '5gbps' in io_lower:
                metadata['storage_interface'] = 'USB 3.0 (5Gbps)'
            elif 'usb 2.0' in io_lower:
                metadata['storage_interface'] = 'USB 2.0'
            elif 'esata' in io_lower:
                if 'usb' in io_lower:
                    metadata['storage_interface'] = 'USB 3.0 & eSATA'
                else:
                    metadata['storage_interface'] = 'eSATA'
            elif 'sata' in io_lower:
                metadata['storage_interface'] = 'SATA'
            else:
                metadata['storage_interface'] = io_str

    enclosure_type = row.get('Form_Factor')
    if pd.notna(enclosure_type):
        et_str = str(enclosure_type).strip()
        if et_str and et_str != 'nan':
            metadata['enclosure_material'] = et_str

    # Tool-free design
    name = metadata.get('name', '')
    if 'tool-free' in name.lower() or 'toolless' in name.lower() or 'tool free' in name.lower():
        metadata['tool_free'] = True

    # =========================================================================
    # CONSOLIDATED DERIVED FIELDS
    # =========================================================================

    category = metadata.get('category', '')

    # --- port_count ---
    if 'kvm' in category:
        _pc = metadata.get('kvm_ports')
        if _pc:
            metadata['port_count'] = _pc
    elif category in ('hub', 'ethernet_switch'):
        _pc = metadata.get('hub_ports')
        if _pc:
            metadata['port_count'] = _pc
    elif category in ('dock', 'multiport_adapter'):
        _tp = metadata.get('total_ports')
        if _tp:
            try:
                metadata['port_count'] = int(float(_tp))
            except (ValueError, TypeError):
                pass
        elif metadata.get('hub_ports'):
            metadata['port_count'] = metadata['hub_ports']
    elif category in ('video_splitter', 'video_switch'):
        # For splitters, prefer the output count from description ("8-Port Splitter" → 8)
        # over raw Ports column which may count input + outputs (9 total)
        _desc = str(metadata.get('description', ''))
        _desc_match = re.search(r'(\d+)[\s-]*[Pp]ort', _desc)
        if _desc_match:
            try:
                _desc_ports = int(_desc_match.group(1))
                metadata['port_count'] = _desc_ports
                # Sync hub_ports so product cards match narrowing
                metadata['hub_ports'] = _desc_ports
                metadata['hub_ports_raw'] = float(_desc_ports)
            except (ValueError, TypeError):
                pass
        if not metadata.get('port_count'):
            for _field in ('total_ports', 'hub_ports'):
                _val = metadata.get(_field)
                if _val:
                    try:
                        metadata['port_count'] = int(float(_val))
                    except (ValueError, TypeError):
                        pass
                    break
        if not metadata.get('port_count'):
            _av = metadata.get('av_output')
            if _av:
                _m = re.search(r'(\d+)', str(_av))
                if _m:
                    try:
                        metadata['port_count'] = int(_m.group(1))
                    except (ValueError, TypeError):
                        pass

    # --- bay_count ---
    _nd = metadata.get('num_drives')
    if _nd:
        metadata['bay_count'] = _nd
    else:
        _total_bays = 0
        for _bay_col in ['Number_of_2.5_Inch_Bays', 'Number_of_3.5_Inch_Bays',
                         'Number_of_External_5.25_Inch_Bays']:
            _bv = row.get(_bay_col)
            if pd.notna(_bv):
                try:
                    _total_bays += int(float(_bv))
                except (ValueError, TypeError):
                    pass
        if _total_bays > 0:
            metadata['bay_count'] = _total_bays

    # --- rack_height_u ---
    _uh = metadata.get('u_height')
    if _uh:
        try:
            _clean = str(_uh).strip().rstrip('Uu').strip()
            metadata['rack_height_u'] = int(float(_clean))
        except (ValueError, TypeError):
            pass

    # --- power_delivery_watts ---
    _max_watts = 0
    for _pf in ['Laptop_Charging_via_Power_Delivery', 'power_delivery',
                'hub_power_delivery', 'power_consumption']:
        _pv = metadata.get(_pf)
        if _pv:
            _wm = re.search(r'(\d+)\s*[wW]', str(_pv))
            if _wm:
                _w = int(_wm.group(1))
                if _w > _max_watts and _w <= 240:
                    _max_watts = _w
    if _max_watts > 0:
        metadata['power_delivery_watts'] = _max_watts

    # --- screen_size_inches ---
    _ss = row.get('Screen_Size')
    if pd.notna(_ss):
        _sm = re.search(r'([\d.]+)', str(_ss).strip())
        if _sm:
            try:
                metadata['screen_size_inches'] = float(_sm.group(1))
            except (ValueError, TypeError):
                pass

    # --- usb_version ---
    if 'hub_usb_version' in metadata:
        metadata['usb_version'] = metadata['hub_usb_version']
    elif 'storage_interface' in metadata and 'usb' in metadata['storage_interface'].lower():
        metadata['usb_version'] = metadata['storage_interface']
