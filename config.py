# 1200 * 910 the standers, pls edit below for test image dim
configuration = {

    "resize": (224, 224),

    "crop_positions": {

        "Q1": {"x1": 772, "y1": 91, "x2": (772 + 360), "y2": (91 + 360), "center": (967, 275), "radius": 143,
               "apply_circular_mask": True},

        "Q2": {"x1": 410, "y1": 90, "x2": (410 + 360), "y2": (90 + 360), "center": (592, 275), "radius": 143,
               "apply_circular_mask": True},

        "Q3": {"x1": 410, "y1": 467, "x2": (410 + 360), "y2": (467 + 360), "center": (592, 652), "radius": 143,
               "apply_circular_mask": True},

        "Q4": {"x1": 771, "y1": 467, "x2": (771 + 360), "y2": (467 + 360), "center": (967, 652), "radius": 143,
               "apply_circular_mask": True},

        "text_panel": {"x1": 10, "y1": 200, "x2": (10 + 315), "y2": (200 + 625)}
    }
}
