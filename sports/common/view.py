from __future__ import annotations

from typing import Tuple, Optional
import cv2
import numpy as np
import numpy.typing as npt

class ViewTransformer:
    def __init__(
        self,
        source: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        ransac_reproj_threshold: float = 3.0,
        confidence: float = 0.995,
        max_iters: int = 5000,
    ) -> None:
        """
        Initialize the ViewTransformer with source and target points.

        Args:
            source: Source points (Nx2) for homography calculation (image coords).
            target: Target points (Nx2) for homography calculation (field coords).
            ransac_reproj_threshold: RANSAC reprojection threshold in pixels (source space).
            confidence: RANSAC confidence.
            max_iters: RANSAC max iterations.

        Raises:
            ValueError: If shapes are invalid or homography can't be computed.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("Source and target points must be Nx2 coordinates.")

        if source.shape[0] < 4:
            raise ValueError(f"Need at least 4 point pairs to compute homography, got {source.shape[0]}.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # Robust solve
        self.m, self.inliers = cv2.findHomography(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    def transform_points(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Transform the given points using the homography matrix.

        Args:
            points: Points (Nx2) to be transformed.

        Returns:
            Transformed points (Nx2).

        Raises:
            ValueError: If points are not Nx2.
        """
        if points.size == 0:
            return points.astype(np.float32)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be Nx2 coordinates.")

        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2).astype(np.float32)

    def transform_image(self, image: npt.NDArray[np.uint8], resolution_wh: Tuple[int, int]) -> npt.NDArray[np.uint8]:
        """
        Transform the given image using the homography matrix.

        Args:
            image: Image to be transformed.
            resolution_wh: (width, height) of the output image.

        Returns:
            Warped image.
        """
        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be either grayscale or color.")
        return cv2.warpPerspective(image, self.m, resolution_wh)