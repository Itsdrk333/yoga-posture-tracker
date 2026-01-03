"""
Utility functions for distance and angle calculations.

This module provides helper functions for computing geometric measurements
commonly used in pose estimation and posture tracking applications.
"""

import math
import numpy as np
from typing import Tuple, Union


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two 2D points.
    
    Args:
        point1: Tuple of (x, y) coordinates for the first point
        point2: Tuple of (x, y) coordinates for the second point
    
    Returns:
        The Euclidean distance between the two points
    
    Example:
        >>> distance = calculate_distance((0, 0), (3, 4))
        >>> print(distance)
        5.0
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_distance_3d(point1: Tuple[float, float, float], 
                          point2: Tuple[float, float, float]) -> float:
    """
    Calculate the Euclidean distance between two 3D points.
    
    Args:
        point1: Tuple of (x, y, z) coordinates for the first point
        point2: Tuple of (x, y, z) coordinates for the second point
    
    Returns:
        The Euclidean distance between the two 3D points
    
    Example:
        >>> distance = calculate_distance_3d((0, 0, 0), (1, 2, 2))
        >>> print(distance)
        3.0
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calculate_angle(point1: Tuple[float, float], 
                   vertex: Tuple[float, float], 
                   point2: Tuple[float, float]) -> float:
    """
    Calculate the angle (in degrees) formed by three points.
    
    The angle is calculated at the vertex point, formed by the rays from
    vertex to point1 and from vertex to point2.
    
    Args:
        point1: Tuple of (x, y) coordinates for the first point
        vertex: Tuple of (x, y) coordinates for the vertex (angle center)
        point2: Tuple of (x, y) coordinates for the second point
    
    Returns:
        The angle in degrees (0-180)
    
    Example:
        >>> angle = calculate_angle((0, 0), (0, 1), (1, 1))
        >>> print(angle)
        45.0
    """
    x1, y1 = point1
    vx, vy = vertex
    x2, y2 = point2
    
    # Vectors from vertex to point1 and point2
    vec1 = np.array([x1 - vx, y1 - vy])
    vec2 = np.array([x2 - vx, y2 - vy])
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate dot product and angle
    dot_product = np.dot(vec1, vec2)
    cos_angle = dot_product / (mag1 * mag2)
    
    # Clamp to [-1, 1] to avoid numerical errors with acos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def calculate_angle_3d(point1: Tuple[float, float, float],
                      vertex: Tuple[float, float, float],
                      point2: Tuple[float, float, float]) -> float:
    """
    Calculate the angle (in degrees) formed by three 3D points.
    
    The angle is calculated at the vertex point, formed by the rays from
    vertex to point1 and from vertex to point2.
    
    Args:
        point1: Tuple of (x, y, z) coordinates for the first point
        vertex: Tuple of (x, y, z) coordinates for the vertex (angle center)
        point2: Tuple of (x, y, z) coordinates for the second point
    
    Returns:
        The angle in degrees (0-180)
    
    Example:
        >>> angle = calculate_angle_3d((0, 0, 0), (0, 0, 1), (1, 0, 1))
        >>> print(angle)
        90.0
    """
    x1, y1, z1 = point1
    vx, vy, vz = vertex
    x2, y2, z2 = point2
    
    # Vectors from vertex to point1 and point2
    vec1 = np.array([x1 - vx, y1 - vy, z1 - vz])
    vec2 = np.array([x2 - vx, y2 - vy, z2 - vz])
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate dot product and angle
    dot_product = np.dot(vec1, vec2)
    cos_angle = dot_product / (mag1 * mag2)
    
    # Clamp to [-1, 1] to avoid numerical errors with acos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def normalize_vector(vector: Tuple[float, float]) -> Tuple[float, float]:
    """
    Normalize a 2D vector to unit length.
    
    Args:
        vector: Tuple of (x, y) components
    
    Returns:
        Normalized vector as a tuple
    
    Example:
        >>> normalized = normalize_vector((3, 4))
        >>> print(normalized)
        (0.6, 0.8)
    """
    x, y = vector
    magnitude = math.sqrt(x ** 2 + y ** 2)
    
    if magnitude == 0:
        return (0.0, 0.0)
    
    return (x / magnitude, y / magnitude)


def normalize_vector_3d(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Normalize a 3D vector to unit length.
    
    Args:
        vector: Tuple of (x, y, z) components
    
    Returns:
        Normalized vector as a tuple
    
    Example:
        >>> normalized = normalize_vector_3d((1, 2, 2))
        >>> print(normalized)
        (0.333..., 0.666..., 0.666...)
    """
    x, y, z = vector
    magnitude = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    if magnitude == 0:
        return (0.0, 0.0, 0.0)
    
    return (x / magnitude, y / magnitude, z / magnitude)


def midpoint(point1: Tuple[float, float], point2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate the midpoint between two 2D points.
    
    Args:
        point1: Tuple of (x, y) coordinates for the first point
        point2: Tuple of (x, y) coordinates for the second point
    
    Returns:
        Midpoint as a tuple of (x, y) coordinates
    
    Example:
        >>> mid = midpoint((0, 0), (10, 10))
        >>> print(mid)
        (5.0, 5.0)
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def midpoint_3d(point1: Tuple[float, float, float], 
                point2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Calculate the midpoint between two 3D points.
    
    Args:
        point1: Tuple of (x, y, z) coordinates for the first point
        point2: Tuple of (x, y, z) coordinates for the second point
    
    Returns:
        Midpoint as a tuple of (x, y, z) coordinates
    
    Example:
        >>> mid = midpoint_3d((0, 0, 0), (10, 10, 10))
        >>> print(mid)
        (5.0, 5.0, 5.0)
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)


def point_to_line_distance(point: Tuple[float, float],
                          line_point1: Tuple[float, float],
                          line_point2: Tuple[float, float]) -> float:
    """
    Calculate the perpendicular distance from a point to a line defined by two points.
    
    Args:
        point: Tuple of (x, y) coordinates of the point
        line_point1: Tuple of (x, y) coordinates for the first point on the line
        line_point2: Tuple of (x, y) coordinates for the second point on the line
    
    Returns:
        The perpendicular distance from the point to the line
    
    Example:
        >>> distance = point_to_line_distance((1, 1), (0, 0), (2, 0))
        >>> print(distance)
        1.0
    """
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    
    # Calculate the distance using the cross product formula
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    
    if denominator == 0:
        return calculate_distance(point, line_point1)
    
    return numerator / denominator


def is_angle_within_range(angle: float, target_angle: float, tolerance: float = 5.0) -> bool:
    """
    Check if an angle is within a specified tolerance of a target angle.
    
    Args:
        angle: The angle to check (in degrees)
        target_angle: The target angle (in degrees)
        tolerance: The acceptable tolerance in degrees (default: 5.0)
    
    Returns:
        True if the angle is within the tolerance range, False otherwise
    
    Example:
        >>> is_angle_within_range(85, 90, tolerance=10)
        True
    """
    difference = abs(angle - target_angle)
    # Handle wrap-around at 0/360 degrees
    difference = min(difference, 360 - difference)
    return difference <= tolerance
