import math
from typing import Tuple

# Type alias for coordinates
Coord = Tuple[float, float]

def calculate_vector_angle(v1: Coord, v2: Coord) -> float:
    """
    Calculates the minimum angle between two 2D vectors in radians (range [0, pi]).

    Args:
        v1 (Coord): The first vector (dx, dy).
        v2 (Coord): The second vector (dx, dy).

    Returns:
        float: The angle in radians between the vectors (0 to pi).
               Returns 0 if either vector is (0,0).
    """
    len1_sq = v1[0]**2 + v1[1]**2
    len2_sq = v2[0]**2 + v2[1]**2

    if len1_sq == 0 or len2_sq == 0:
        return 0.0 # Angle with zero vector is undefined, return 0

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    len1 = math.sqrt(len1_sq)
    len2 = math.sqrt(len2_sq)

    # Clamp cos_theta to [-1, 1] due to potential floating point inaccuracies
    cos_theta = max(-1.0, min(1.0, dot / (len1 * len2)))

    angle_rad = math.acos(cos_theta) # This gives angle in [0, pi]

    return angle_rad

# --- Line Segment Intersection Helpers ---
# Based on standard geometric algorithms (e.g., Cormen et al.)
# Used for planarity checks during generation [Implied by Planar Graph Goal, Sec 3.1]

def on_segment(p: Coord, q: Coord, r: Coord) -> bool:
    """
    Given three collinear points p, q, r, check if point q lies on line segment 'pr'.

    Args:
        p (Coord): First point of the segment.
        q (Coord): Point to check.
        r (Coord): Second point of the segment.

    Returns:
        bool: True if q lies on segment pr, False otherwise.
    """
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def orientation(p: Coord, q: Coord, r: Coord) -> int:
    """
    Find the orientation of the ordered triplet (p, q, r).

    Args:
        p (Coord): First point.
        q (Coord): Second point.
        r (Coord): Third point.

    Returns:
        int: 0 if p, q, r are collinear,
             1 if orientation is clockwise,
             2 if orientation is counterclockwise.
    """
    # Using cross-product approach
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    if math.isclose(val, 0): # Use math.isclose for float comparison
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

def do_intersect(p1: Coord, q1: Coord, p2: Coord, q2: Coord) -> bool:
    """
    Check if line segment 'p1q1' and line segment 'p2q2' intersect.

    Args:
        p1 (Coord): Start point of the first segment.
        q1 (Coord): End point of the first segment.
        p2 (Coord): Start point of the second segment.
        q2 (Coord): End point of the second segment.

    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    # Find the four orientations needed for general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # --- General Case ---
    # If orientations o1/o2 differ AND o3/o4 differ, the segments cross each other.
    if o1 != o2 and o3 != o4:
        # Additionally check that the intersection is not *only* at an endpoint
        # if the endpoint is shared. This basic check assumes non-degenerate cases.
        # A more robust check might be needed for complex scenarios, but this
        # covers the standard intersection case.
        return True

    # --- Special Cases (Collinearity) ---
    # Check if endpoints of one segment lie on the other segment if they are collinear.

    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1): return True
    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1): return True
    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2): return True
    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2): return True

    # Segments do not intersect
    return False
