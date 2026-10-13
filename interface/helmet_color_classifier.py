"""
Helmet Color Classifier & Occupation Mapping Module for SiteSentinel.
Classifies hardhat colors in real-time and evaluates cross-checks against worker occupation roles.

Color to Occupation Mapping (Industry Standard Safety Helmet Coding):
- Yellow: Labourer, Heavy-duty operations, & Construction tasks
- White: Managers, Engineers, Supervisors, Foreman, & Site Visitors (Grey combined into White)
- Red: Firefighters
- Brown: Welders, high heat operations
- Blue: Electricians and Technical operators
- Green: Safety officers
- Pink: Female workers* / Additional helmet
"""
import cv2
import numpy as np

# Official mapping between Occupation and Helmet Color
OCCUPATION_TO_COLOR = {
    "Labourer / Construction": "Yellow",
    "Manager / Supervisor / Visitor": "White",
    "Firefighter": "Red",
    "Welder / High Heat": "Brown",
    "Electrician / Technical": "Blue",
    "Safety Officer": "Green",
    "Female Worker / Additional": "Pink"
}

# Synonyms and partial matches
OCCUPATION_SYNONYMS = {
    "labourer": "Yellow",
    "laborer": "Yellow",
    "construction": "Yellow",
    "heavy-duty": "Yellow",
    "manager": "White",
    "engineer": "White",
    "supervisor": "White",
    "foreman": "White",
    "visitor": "White",
    "site visitor": "White",
    "firefighter": "Red",
    "welder": "Brown",
    "welding": "Brown",
    "high heat": "Brown",
    "electrician": "Blue",
    "technical": "Blue",
    "safety officer": "Green",
    "safety": "Green",
    "female": "Pink",
    "additional": "Pink"
}

COLOR_DISPLAY_METADATA = {
    "Yellow": {
        "bgr": (0, 220, 240),
        "hex": "#eab308",
        "description": "Labourer, Heavy-duty operations, & Construction tasks"
    },
    "White": {
        "bgr": (240, 240, 240),
        "hex": "#f8fafc",
        "description": "Managers, Engineers, Supervisors, Foreman, & Site Visitors"
    },
    "Red": {
        "bgr": (30, 30, 225),
        "hex": "#ef4444",
        "description": "Firefighters"
    },
    "Brown": {
        "bgr": (25, 60, 120),
        "hex": "#8b5a2b",
        "description": "Welders, high heat operations"
    },
    "Blue": {
        "bgr": (220, 110, 20),
        "hex": "#3b82f6",
        "description": "Electricians and Technical operators"
    },
    "Green": {
        "bgr": (35, 180, 45),
        "hex": "#22c55e",
        "description": "Safety officers"
    },
    "Pink": {
        "bgr": (180, 80, 240),
        "hex": "#ec4899",
        "description": "Female workers* / Additional helmet"
    }
}


def get_expected_color_for_occupation(occupation: str) -> str:
    """Returns the expected helmet color for a given occupation string."""
    if not occupation:
        return "Yellow"
    clean = occupation.strip()
    if clean in OCCUPATION_TO_COLOR:
        return OCCUPATION_TO_COLOR[clean]

    clean_lower = clean.lower()
    for key, color in OCCUPATION_SYNONYMS.items():
        if key in clean_lower:
            return color

    return "Yellow"


def is_color_matching_occupation(detected_color: str, occupation: str) -> bool:
    """
    Checks if the detected helmet color matches the worker's occupation.
    Note: Both White and Grey are treated as White.
    """
    if not detected_color or detected_color == "Unknown":
        return True  # If color could not be determined with confidence, do not hard-fail
    
    # Normalize grey to white
    det_norm = "White" if detected_color.lower() in ["white", "grey", "gray"] else detected_color.capitalize()
    expected = get_expected_color_for_occupation(occupation)
    return det_norm.lower() == expected.lower()


def classify_helmet_color(crop_bgr: np.ndarray) -> tuple[str, float]:
    """
    Analyzes a hardhat crop (BGR image) in HSV color space and determines its dominant color.
    Returns: (detected_color: str, confidence: float)
    Supported colors: Yellow, White (including Grey), Red, Brown, Blue, Green, Pink.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown", 0.0

    h, w = crop_bgr.shape[:2]
    # Sample upper-central dome of the helmet to avoid hair, neck, and background
    y1, y2 = int(h * 0.12), int(h * 0.80)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    
    dome = crop_bgr[y1:y2, x1:x2]
    if dome.size == 0:
        dome = crop_bgr

    # Convert to HSV
    hsv = cv2.cvtColor(dome, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Exclude extreme shadows/dark hair/black pixels
    valid_mask = (V > 32)
    total_valid = np.count_nonzero(valid_mask)
    if total_valid < 25:
        return "Unknown", 0.0

    counts = {}

    # 1. White / Grey (Neutral low saturation, combined into White)
    white_mask = valid_mask & (S < 45) & (V > 65)
    counts["White"] = np.count_nonzero(white_mask)

    # 2. Yellow (Warm high hue and brightness)
    yellow_mask = valid_mask & (S >= 45) & (H >= 20) & (H <= 38) & (V >= 90)
    counts["Yellow"] = np.count_nonzero(yellow_mask)

    # 3. Blue
    blue_mask = valid_mask & (S >= 40) & (H >= 90) & (H <= 135)
    counts["Blue"] = np.count_nonzero(blue_mask)

    # 4. Green
    green_mask = valid_mask & (S >= 40) & (H >= 39) & (H <= 89)
    counts["Green"] = np.count_nonzero(green_mask)

    # 5. Pink (Magenta)
    pink_mask = valid_mask & (S >= 40) & (H >= 142) & (H <= 170) & (V >= 95)
    counts["Pink"] = np.count_nonzero(pink_mask)

    # 6. Brown (Dark warm orange/brown tone)
    brown_mask = valid_mask & (S >= 38) & (H >= 8) & (H <= 24) & (V >= 32) & (V < 145)
    counts["Brown"] = np.count_nonzero(brown_mask)

    # 7. Red (Hue wraps around 0 and 180, high saturation and value)
    red_mask = valid_mask & (S >= 55) & ((H < 10) | (H > 170)) & (V >= 90)
    counts["Red"] = np.count_nonzero(red_mask)

    best_color = max(counts, key=counts.get)
    best_count = counts[best_color]
    conf = round(best_count / float(total_valid), 2)

    # Require minimum coverage threshold for definitive color
    if conf < 0.20:
        # Fall back to White if desaturated
        if counts["White"] > 0:
            return "White", round(counts["White"] / float(total_valid), 2)
        return "Unknown", conf

    return best_color, conf
