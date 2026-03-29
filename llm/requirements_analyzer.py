"""
Requirements Analyzer - Extract structured requirements from complex queries.

Uses LLM to understand the full context of what the user needs:
- Host device type (laptop, desktop, tablet)
- Display requirements (monitor count, resolution, refresh rate)
- Peripheral needs (what devices they mention)
- Charging requirements (host vs downstream devices)
- Questions they asked (that need direct answers)

This enables consultative responses that directly address user needs.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from core.openai_client import get_openai_client
from core.api_retry import RetryHandler, DEFAULT_OPENAI_RETRY


@dataclass
class SolutionComponent:
    """A component of a multi-product solution."""
    role: str = ""              # Human-readable: "Video Switching", "Signal Conversion", "Connectivity"
    category: str = ""          # Product category: "video_switch", "adapter", "cable"
    connector_from: Optional[str] = None  # SOURCE device output (laptop/computer port)
    connector_to: Optional[str] = None    # DESTINATION device input (monitor/projector port)
    length_ft: Optional[float] = None     # Cable length needed (if applicable)
    priority: int = 2           # 1=essential, 2=recommended, 3=optional
    reason: str = ""            # Why this component is needed
    quantity: int = 1           # How many of this component needed


@dataclass
class UserRequirements:
    """Structured requirements extracted from a complex query."""

    # Host device info
    host_device: Optional[str] = None  # "Mac Studio M2 Max", "MacBook Pro 14"
    host_type: Optional[str] = None    # "desktop", "laptop", "tablet"

    # Display needs
    monitor_count: Optional[int] = None
    resolution: Optional[str] = None   # "4K", "1080p", "1440p"
    refresh_rate: Optional[int] = None  # 60, 144, etc.

    # Peripherals mentioned
    peripherals: List[str] = field(default_factory=list)
    # e.g., ["external hard drives", "SD card reader", "keyboard", "mouse"]

    # Charging needs
    charging_device: Optional[str] = None  # "iPad Pro", "laptop", etc.
    charging_type: str = "none"  # "host", "downstream", "none"
    # host = charge the main computer via dock
    # downstream = charge peripherals like iPad, phone via dock's ports

    # Questions the user explicitly asked
    questions: List[str] = field(default_factory=list)
    # e.g., ["will one dock handle all of this?", "what's the best solution?"]

    # Computed port needs
    min_usb_a_ports: int = 0
    min_usb_c_ports: int = 0

    # Raw extraction metadata
    confidence: float = 0.0
    reasoning: str = ""

    # Multi-product solution components
    solution_components: List[SolutionComponent] = field(default_factory=list)
    is_multi_product_solution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "host_device": self.host_device,
            "host_type": self.host_type,
            "monitor_count": self.monitor_count,
            "resolution": self.resolution,
            "refresh_rate": self.refresh_rate,
            "peripherals": self.peripherals,
            "charging_device": self.charging_device,
            "charging_type": self.charging_type,
            "questions": self.questions,
            "min_usb_a_ports": self.min_usb_a_ports,
            "min_usb_c_ports": self.min_usb_c_ports,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "solution_components": [
                {
                    "role": c.role,
                    "category": c.category,
                    "connector_from": c.connector_from,
                    "connector_to": c.connector_to,
                    "length_ft": c.length_ft,
                    "priority": c.priority,
                    "reason": c.reason,
                    "quantity": c.quantity,
                }
                for c in self.solution_components
            ],
            "is_multi_product_solution": self.is_multi_product_solution,
        }


# System prompt for requirements extraction
REQUIREMENTS_PROMPT = """You are analyzing a customer support query for StarTech.com (computer connectivity products).

Extract the user's REQUIREMENTS and QUESTIONS from their query. Focus on understanding what they need, not just what product category they want.

## What to Extract

1. **Host Device** - What computer/device are they connecting FROM?
   - host_device: Specific model if mentioned ("Mac Studio M2 Max", "Dell XPS 15")
   - host_type: "desktop", "laptop", or "tablet"
   - IMPORTANT: Mac Studio, Mac Pro, iMac = desktop (not laptop!)

2. **Display Requirements**
   - monitor_count: How many monitors they want to connect
   - resolution: "4K", "1080p", "1440p", "8K" if mentioned
   - refresh_rate: Hz value if mentioned (60, 144, etc.)

3. **Peripherals** - List ALL devices they mention needing to connect:
   - Storage: "external hard drives", "SSD", "USB drives"
   - Input: "keyboard", "mouse", "trackpad"
   - Media: "SD card reader", "webcam", "microphone"
   - Other: Any device they explicitly mention

4. **Charging Needs** - CRITICAL distinction:
   - charging_device: What device needs charging
   - charging_type:
     - "host" = They want the dock to charge their laptop/computer
     - "downstream" = They want to charge OTHER devices (iPad, phone) via dock ports
     - "none" = No charging mentioned
   - IMPORTANT: If they have a DESKTOP (Mac Studio, tower PC), they don't need host charging!
     If they mention charging iPad/phone with a desktop, that's "downstream" charging.

5. **Questions** - What did they explicitly ASK?
   - Look for question marks and phrases like "will this...", "can I...", "what's the best..."
   - Extract the actual questions they want answered

6. **Port Counts** - Estimate minimum ports needed:
   - min_usb_a_ports: Count USB-A devices mentioned (keyboards, drives usually USB-A)
   - min_usb_c_ports: Count USB-C devices (newer devices, charging needs)

7. **Solution Components** - Does this require MULTIPLE product types?
   Analyze if the user's setup needs multiple products from different categories working together.

   Common multi-product scenarios:
   - Source switching: Multiple devices → one display (needs switch + cables)
   - Signal conversion: When source port type ≠ monitor port type (ALWAYS needs adapter)
     Examples: HDMI source → DisplayPort monitor = needs HDMI-to-DP adapter
              DisplayPort source → HDMI monitor = needs DP-to-HDMI adapter
              USB-C source → HDMI monitor = needs USB-C-to-HDMI adapter
     ALWAYS include a "Signal Conversion" component when ports don't match!

   ⚠️ SIGNAL FLOW DIRECTION - CRITICAL:
   - connector_from = OUTPUT connector of the SOURCE device (laptop, computer, gaming console)
   - connector_to = INPUT connector of the DESTINATION device (monitor, projector, TV)

   When user says "laptop has VGA" and "projector has HDMI":
   - VGA is the laptop's OUTPUT (connector_from)
   - HDMI is the projector's INPUT (connector_to)
   - Adapter needed: VGA-to-HDMI (NOT HDMI-to-VGA!)

   Think of it as: Signal flows FROM source device TO destination device
   - "VGA laptop to HDMI projector" → connector_from="VGA", connector_to="HDMI"
   - "USB-C laptop to DisplayPort monitor" → connector_from="USB-C", connector_to="DisplayPort"
   - "DisplayPort computer to HDMI TV" → connector_from="DisplayPort", connector_to="HDMI"
   - Distance/length mentioned: ALWAYS needs cables with appropriate length
   - KVM setup: Share keyboard/mouse/peripherals between computers (needs KVM + cables)
   - Multi-monitor from laptop: May need dock + cables

   IMPORTANT distinctions:
   - Use "video_switch" for switching VIDEO sources only (laptop + gaming console → monitor)
   - Use "kvm_switch" ONLY if user wants to share keyboard/mouse between computers
   - If user mentions distance/length (feet, meters, "far away", "across the room", etc.),
     ALWAYS include a "Connectivity" component with category="cable" and appropriate length_ft.
     Cables are needed to physically connect devices - don't assume they're included elsewhere.

   For EACH component needed, specify:
   - role: What function it serves ("Video Switching", "Signal Conversion", "Connectivity")
   - category: Product category ("video_switch", "adapter", "cable", "kvm_switch", "dock", "hub")
   - connector_from/to: Connection types (e.g., "HDMI", "DisplayPort", "USB-C")
     ⚠️ For cables (category="cable"), connector_from is REQUIRED to find the right cable type!
     - HDMI cables: connector_from="HDMI"
     - DisplayPort cables: connector_from="DisplayPort"
     - USB-C cables: connector_from="USB-C"
     - Ethernet cables: connector_from="Ethernet" or "RJ45"
   - length_ft: Cable length if distance mentioned (convert meters to feet: 1m ≈ 3.3ft)
   - priority: 1=must have, 2=recommended, 3=nice to have
   - reason: Brief explanation of why this component is needed
   - quantity: How many of this component needed (default 1)

   QUANTITY CALCULATION RULES:
   - For switches/KVMs: usually 1
   - For adapters: usually 1 per signal path
   - For cables connecting source devices to a switch: quantity = number of source devices
     Example: laptop + PS5 → switch = 2 cables needed (one from each source to the switch)
   - Count source devices mentioned and set cable quantity accordingly

   ⚠️ KVM/MULTI-COMPUTER SETUPS - Create SEPARATE cable components:
   KVM switches require MULTIPLE cable types. Do NOT bundle into one "Connectivity" component.

   For KVM with N computers and M monitors, you need:
   1. "Connectivity (Input Video)" - Video cables from computers → KVM
      - quantity = N (one cable per computer)
      - connector_from = video type (HDMI, DisplayPort, etc.)
   2. "Connectivity (Output Video)" - Video cables from KVM → monitors
      - quantity = M (one cable per monitor)
      - connector_from = video type
   3. "Connectivity (USB Control)" - USB cables for keyboard/mouse sharing
      - quantity = N (one cable per computer)
      - connector_from = "USB-A" or "USB-B"

   Example calculation for 3 computers + 2 monitors:
   - Input video cables: 3 (computers → KVM)
   - Output video cables: 2 (KVM → monitors)
   - USB control cables: 3 (for peripheral switching)
   - TOTAL: 8 cables minimum

   IMPORTANT: Usually create ONE component per role, BUT for Signal Conversion:
   - If multiple SOURCE devices have DIFFERENT output connectors, create SEPARATE adapter entries
   - Example: Conference room with USB-C, VGA, and HDMI laptops going to one HDMI projector:
     - Signal Conversion (VGA): VGA→HDMI adapter (for VGA laptops)
     - Signal Conversion (USB-C): USB-C→HDMI adapter (for USB-C laptops)
     - HDMI laptops connect directly (no adapter needed)
   - For cables: use quantity field (ONE Connectivity component with quantity=3 for 3 cables)
   - For switches: usually 1

   Set is_multi_product_solution=true if ANY of these are true:
   - 2+ different product categories are needed (e.g., switch + cables)
   - Multiple solution_components with DIFFERENT connector types (e.g., USB-C cables AND DisplayPort cables)
   - Daisy-chain setups (ALWAYS multi-product: primary cable + chain cables are different types)
   - KVM setups (ALWAYS multi-product: video cables + USB control cables)

   Example for conference room with multiple laptop types:
   User says: "Conference room, laptops have USB-C, HDMI, or VGA, projector is HDMI, 40 feet away"
   Solution components:
   - {"role": "Signal Conversion (VGA)", "category": "adapter", "connector_from": "VGA", "connector_to": "HDMI", "reason": "For VGA laptops"}
   - {"role": "Signal Conversion (USB-C)", "category": "adapter", "connector_from": "USB-C", "connector_to": "HDMI", "reason": "For USB-C laptops"}
   - {"role": "Connectivity", "category": "cable", "connector_from": "HDMI", "length_ft": 40, "reason": "HDMI cable to projector"}

   Example for KVM setup (3 computers, 2 monitors, shared peripherals):
   User says: "3 computers, 2 monitors, switch between them, share keyboard/mouse"
   Solution components:
   - {"role": "Video Switching", "category": "kvm_switch", "quantity": 1, "reason": "KVM for switching 3 computers on 2 monitors"}
   - {"role": "Connectivity (Input Video)", "category": "cable", "connector_from": "HDMI", "quantity": 3, "reason": "Video from each computer to KVM"}
   - {"role": "Connectivity (Output Video)", "category": "cable", "connector_from": "HDMI", "quantity": 2, "reason": "Video from KVM to each monitor"}
   - {"role": "Connectivity (USB Control)", "category": "cable", "connector_from": "USB-A", "quantity": 3, "reason": "USB for keyboard/mouse control of each computer"}

   ⚠️ DAISY-CHAIN MONITOR SETUPS - Requires SPECIFIC cable combinations:
   Daisy-chaining connects monitors in series: Computer → Monitor1 → Monitor2 → Monitor3
   This requires DisplayPort MST (Multi-Stream Transport) or Thunderbolt.

   For daisy-chaining N monitors:
   1. "Connectivity (Primary)" - Cable from computer to FIRST monitor
      - quantity = 1
      - connector_from = computer's output (USB-C or DisplayPort)
      - connector_to = first monitor's input (usually USB-C for hub monitors, or DisplayPort)
      - reason = "Connects computer to first monitor in chain"

   2. "Connectivity (Chain)" - DP-to-DP cables BETWEEN monitors
      - quantity = N-1 (one less than monitor count)
      - connector_from = "DisplayPort"
      - connector_to = "DisplayPort"
      - reason = "Connects monitors in daisy-chain (DP Out → DP In)"

   Example for 3 daisy-chained monitors (USB-C hub monitors like Dell P2725HE):
   User says: "I need cables to daisy-chain 3 monitors"
   Solution components:
   - {"role": "Connectivity (Primary)", "category": "cable", "connector_from": "USB-C", "connector_to": "USB-C", "quantity": 1, "reason": "Computer to first monitor (uses USB-C hub features)"}
   - {"role": "Connectivity (Chain)", "category": "cable", "connector_from": "DisplayPort", "connector_to": "DisplayPort", "quantity": 2, "reason": "DP Out to DP In between monitors for daisy-chain"}

   IMPORTANT: The daisy-chain cables MUST be DisplayPort-to-DisplayPort (not USB-C to DP).
   Monitors use their DP OUT port to chain to the next monitor's DP IN port.

Respond ONLY with valid JSON:
{
    "host_device": "string or null",
    "host_type": "desktop|laptop|tablet|null",
    "monitor_count": "number or null",
    "resolution": "string or null",
    "refresh_rate": "number or null",
    "peripherals": ["list of devices mentioned"],
    "charging_device": "string or null",
    "charging_type": "host|downstream|none",
    "questions": ["list of questions they asked"],
    "min_usb_a_ports": "number",
    "min_usb_c_ports": "number",
    "solution_components": [
        {
            "role": "string - function name",
            "category": "video_switch|adapter|cable|kvm_switch|dock|hub",
            "connector_from": "string or null",
            "connector_to": "string or null",
            "length_ft": "number or null",
            "priority": 1|2|3,
            "reason": "why needed",
            "quantity": "number (default 1)"
        }
    ],
    "is_multi_product_solution": true|false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of what user needs"
}"""


class RequirementsAnalyzer:
    """
    Analyzes complex queries to extract structured user requirements.

    Used for generating consultative responses that directly address
    what the user needs, not just list products.
    """

    def __init__(self):
        self.client = get_openai_client()
        self.retry_handler = RetryHandler(DEFAULT_OPENAI_RETRY)

    def analyze(
        self,
        query: str,
        debug_lines: Optional[List[str]] = None
    ) -> Optional[UserRequirements]:
        """
        Analyze a query to extract structured requirements.

        Args:
            query: The user's natural language query
            debug_lines: Optional list to append debug info

        Returns:
            UserRequirements if extraction succeeds, None on failure
        """
        if not self.client:
            if debug_lines is not None:
                debug_lines.append("⚠️ REQUIREMENTS: No OpenAI client")
            return None

        try:
            result = self.retry_handler.execute(
                lambda: self._call_openai(query),
                fallback=None
            )

            if result and debug_lines is not None:
                debug_lines.append(f"🎯 REQUIREMENTS: {result.reasoning}")
                if result.questions:
                    debug_lines.append(f"   Questions: {result.questions}")
                if result.host_type:
                    debug_lines.append(f"   Host: {result.host_device} ({result.host_type})")
                if result.is_multi_product_solution:
                    components_summary = [f"{c.role}({c.category})" for c in result.solution_components]
                    debug_lines.append(f"🔧 MULTI-PRODUCT: {', '.join(components_summary)}")

            return result

        except Exception as e:
            if debug_lines is not None:
                debug_lines.append(f"⚠️ REQUIREMENTS ERROR: {type(e).__name__}")
            return None

    def _call_openai(self, query: str) -> Optional[UserRequirements]:
        """Make the OpenAI API call."""
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-nano"),
            messages=[
                {"role": "system", "content": REQUIREMENTS_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_completion_tokens=800,  # Increased for solution_components
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)

        # Parse solution_components
        solution_components = []
        for comp in data.get("solution_components", []):
            solution_components.append(SolutionComponent(
                role=comp.get("role", ""),
                category=comp.get("category", ""),
                connector_from=comp.get("connector_from"),
                connector_to=comp.get("connector_to"),
                length_ft=comp.get("length_ft"),
                priority=comp.get("priority", 2),
                reason=comp.get("reason", ""),
                quantity=comp.get("quantity", 1),
            ))

        return UserRequirements(
            host_device=data.get("host_device"),
            host_type=data.get("host_type"),
            monitor_count=data.get("monitor_count"),
            resolution=data.get("resolution"),
            refresh_rate=data.get("refresh_rate"),
            peripherals=data.get("peripherals", []),
            charging_device=data.get("charging_device"),
            charging_type=data.get("charging_type", "none"),
            questions=data.get("questions", []),
            min_usb_a_ports=data.get("min_usb_a_ports", 0),
            min_usb_c_ports=data.get("min_usb_c_ports", 0),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
            solution_components=solution_components,
            is_multi_product_solution=data.get("is_multi_product_solution", False),
        )


# Module-level instance for convenience
_analyzer: Optional[RequirementsAnalyzer] = None


def analyze_requirements(
    query: str,
    debug_lines: Optional[List[str]] = None
) -> Optional[UserRequirements]:
    """
    Convenience function to analyze requirements from a query.

    Args:
        query: User's natural language query
        debug_lines: Optional debug output list

    Returns:
        UserRequirements or None
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = RequirementsAnalyzer()
    return _analyzer.analyze(query, debug_lines)
