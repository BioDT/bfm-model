import torch

EARTH_RADIUS = 6378137 / 1000  # Radius of the earth in KMs


def compute_polygon_area(polygon: torch.Tensor) -> torch.Tensor:
    """Compute the area of a polygon specified by latitudes and longitudes in degrees.

    Args:
        polygon (torch.Tensor): Polygon coordinates. Shape: [..., n, 2] where:
            ... represents optional batch dimensions
            n is the number of points of the polygon
            2 represents [latitude, longitude] in degrees
            The polygon does not need to be closed.

    Returns:
        torch.Tensor: Area in square kilometers. Shape: [...]

    Note:
        This function uses the shoelace formula (also known as the surveyor's formula)
        to calculate the area of the polygon on a sphere.
    """
    # Close the polygon by adding the first point to the end
    closed_polygon = torch.cat((polygon, polygon[..., :1, :]), dim=-2)  # shape: [..., n+1, 2]

    # Initialize area tensor
    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)  # shape: [...]

    num_points = closed_polygon.shape[-2]  # Number of points including the duplicated first point

    # Vectorized computation instead of loop
    if num_points > 2:
        # Extract latitudes and longitudes
        lats = closed_polygon[..., :, 0]  # shape: [..., n+1]
        lons = closed_polygon[..., :, 1]  # shape: [..., n+1]

        # Compute differences in longitude
        lon_diff = torch.deg2rad(lons[..., 1:] - lons[..., :-1])  # shape: [..., n]

        # Compute average latitude for each edge
        lat_avg = torch.deg2rad(0.5 * (lats[..., 1:] + lats[..., :-1]))  # shape: [..., n]

        # Apply shoelace formula
        area = torch.sum(lon_diff * torch.sin(lat_avg), dim=-1)  # shape: [...]

    # Scale area by Earth's radius squared and take absolute value
    area = torch.abs(area * (EARTH_RADIUS**2) / 2)

    return area


def expand_matrix_with_interpolation(matrix: torch.Tensor) -> torch.Tensor:
    """Expand matrix by adding one row and one column to each side, using
    linear interpolation.

    Args:
        matrix: Matrix to expand. Shape: [H, W]

    Returns:
        Expanded matrix with two extra rows and two extra columns. Shape: [H+2, W+2]

    Note:
        This function uses linear extrapolation to add new rows and columns.
    """
    # Add top and bottom rows using linear extrapolation
    # shape: [H+2, W]
    matrix_expanded = torch.cat(
        (
            2 * matrix[0:1] - matrix[1:2],  # Extrapolate top row
            matrix,
            2 * matrix[-1:] - matrix[-2:-1],  # Extrapolate bottom row
        ),
        dim=0,
    )

    # Add left and right columns using linear extrapolation
    # shape: [H+2, W+2]
    matrix_expanded = torch.cat(
        (
            2 * matrix_expanded[:, 0:1] - matrix_expanded[:, 1:2],  # Extrapolate left column
            matrix_expanded,
            2 * matrix_expanded[:, -1:] - matrix_expanded[:, -2:-1],  # Extrapolate right column
        ),
        dim=1,
    )

    return matrix_expanded


def compute_patch_areas(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Compute areas of non-intersecting patches on the Earth defined by latitude and longitude matrices.

    Args:
        lat: Latitude matrix. Must be decreasing along rows. Shape: [H, W]
        lon: Longitude matrix. Must be increasing along columns. Shape: [H, W]

    Returns:
        Areas of patches in square kilometers. Shape: [H-1, W-1]

    Note:
        This function divides the Earth into patches using grid points as patch centers.
        Patch vertices are calculated as midpoints between adjacent grid points.

    """
    if not (lat.dim() == lon.dim() == 2):
        raise ValueError("Latitude and Longitude must both be 2D tensors.")
    if lat.shape != lon.shape:
        raise ValueError("Latitude and Longitude must have the same shape.")

    # Check that the latitude matrix is decreasing along rows
    if not torch.all(lat[1:] - lat[:-1] <= 0):
        raise ValueError("Latitude must be decreasing along rows.")

    # Check that the longitude matrix is increasing along columns
    if not torch.all(lon[:, 1:] - lon[:, :-1] >= 0):
        raise ValueError("Longitude must be increasing along columns.")

    # Enlarge the latitude and longitude matrices for midpoint computation
    lat_expanded = expand_matrix_with_interpolation(lat)  # shape: [H+2, W+2]
    lon_expanded = expand_matrix_with_interpolation(lon)  # shape: [H+2, W+2]

    # Clamp latitudes to valid range (-90 to 90 degrees), this is important for the symmetry of the resulting areas
    lat_expanded = torch.clamp(lat_expanded, -90, 90)

    # Calculate midpoints between entries in lat/lon, this is to ensure symmetry of the resulting areas
    lat_midpoints = 0.25 * (
        lat_expanded[:-1, :-1] + lat_expanded[:-1, 1:] + lat_expanded[1:, :-1] + lat_expanded[1:, 1:]
    )  # shape: [H+1, W+1]
    lon_midpoints = 0.25 * (
        lon_expanded[:-1, :-1] + lon_expanded[:-1, 1:] + lon_expanded[1:, :-1] + lon_expanded[1:, 1:]
    )  # shape: [H+1, W+1]

    # Create polygons for each patch, each polygon is defined by four vertices (top-left, top-right, bottom-right, bottom-left)
    polygons = torch.stack(
        [
            torch.stack((lat_midpoints[1:, :-1], lon_midpoints[1:, :-1]), dim=-1),  # Top-left
            torch.stack((lat_midpoints[1:, 1:], lon_midpoints[1:, 1:]), dim=-1),  # Top-right
            torch.stack((lat_midpoints[:-1, 1:], lon_midpoints[:-1, 1:]), dim=-1),  # Bottom-right
            torch.stack((lat_midpoints[:-1, :-1], lon_midpoints[:-1, :-1]), dim=-1),  # Bottom-left
        ],
        dim=-2,
    )  # shape: [H, W, 4, 2], last dimensions being 4 and 2 because we have 4 vertices and 2 coordinates (lat, lon)

    # Compute areas of polygons
    return compute_polygon_area(polygons)
