import numpy as np
from scipy.ndimage import gaussian_filter


class ImageDataAmputation:
    """
    Unified class for generating missing data in 2D/3D medical images using
    different mechanisms: MCAR (Missing Completely At Random), MAR (Missing At Random),
    and MNAR (Missing Not At Random).

    All methods support both 2D and 3D images and return:
    - Original normalized image
    - Image with missing values (NaN)
    - Binary mask indicating missing pixels (1 = missing, 0 = present)
    """

    def __init__(self, missing_rate: float):
        """
        Initialize the amputation generator.

        Parameters
        ----------
        missing_rate : float
            Target missing data rate (probability). Must be between 0 and 1.
            Example: 0.2 means approximately 20% missing data.
        """
        self.missing_rate = missing_rate

    def _normalize_and_prepare(self, x_data: np.ndarray) -> tuple:
        """
        Internal helper to normalize images and handle 2D/3D shapes.
        Converts to float32 in [0, 1] range and ensures 4D shape (batch, height, width, channels).

        Returns
        -------
        normalized_data : np.ndarray
            Normalized image in range [0, 1]
        foreground_mask : np.ndarray
            Boolean mask identifying foreground pixels (non-background)
        original_shape : tuple
            Original shape for reference
        """
        original_shape = x_data.shape

        # Normalize to [0, 1]
        x_data = x_data.astype("float32") / 255.0

        # Foreground mask: pixels with significant intensity
        foreground_mask = np.any(x_data >= 0.0, axis=0)    

        return x_data, foreground_mask, original_shape

    def _apply_mask(self, x_data: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Internal helper to apply missing mask to data and mark missing values as NaN.

        Parameters
        ----------
        x_data : np.ndarray
            Normalized image data
        missing_mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Returns
        -------
        x_data_missing : np.ndarray
            Data with NaN values where mask indicates missing
        """
        x_data_md = x_data * (~missing_mask.astype(bool)).astype(int) + -1.0 * missing_mask
        x_data_md[x_data_md == -1] = np.nan
        return x_data_md

    # ────────────────────────────────────────────────────────────────────────
    # MCAR: Missing Completely At Random
    # ────────────────────────────────────────────────────────────────────────

    def generate_mcar_uniform(self, x_data: np.ndarray) -> tuple:
        """
        Generate uniformly random missing pixels (MCAR mechanism).

        Each pixel has the same probability of being missing, independent of its
        intensity or location. Missing values are limited to foreground regions.

        **Recommended for:**
        - Sensor noise modeling
        - Random pixel failures
        - General baseline comparison

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)
            Can be uint8 (0-255) or float

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with randomly placed NaN values
        mask : np.ndarray
            Binary mask (1 = missing pixel, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.1)
        >>> original, missing, mask = amputer.generate_mcar_uniform(image)
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        num_channels = x_data.shape[-1]

        # Generate random binary mask with uniform probability
        missing_mask_2d = np.random.choice(
            [0, 1],
            size=(x_data.shape[0], x_data.shape[1], x_data.shape[2]),
            p=[1 - self.missing_rate, self.missing_rate],
        )

        # Expand to all channels and limit to foreground
        missing_mask_limited = missing_mask_2d * foreground_mask
        missing_mask = np.stack(
            (missing_mask_limited,) * num_channels, axis=-1
        ).astype(np.float32)

        x_data_md = self._apply_mask(x_data, missing_mask)

        return x_data, x_data_md, missing_mask

    def generate_mcar_dead_pixels(
        self,
        x_data: np.ndarray,
        p_single: float = 0.003,
        p_cluster: float = 0.001,
        cluster_size: int = 3,
    ) -> tuple:
        """
        Simulate dead pixels and small clusters (sensor defects, burnt pixels).

        Models hardware failures: individual pixel drop-outs and small rectangular
        clusters (burnt columns/rows). Morphology: scattered points and small patches.

        **Recommended for:**
        - Sensor malfunction simulation
        - Hardware-induced artifacts
        - Detector defect modeling

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)
        p_single : float, optional
            Probability of individual dead pixel. Default: 0.003
        p_cluster : float, optional
            Probability of cluster occurrence. Default: 0.001
        cluster_size : int, optional
            Maximum cluster size (pixels). Default: 3

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with dead pixels/clusters replaced by NaN
        mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.05)
        >>> original, missing, mask = amputer.generate_mcar_dead_pixels(
        ...     image, p_single=0.005, p_cluster=0.002
        ... )
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        num_channels = x_data.shape[-1]
        batch, height, width = x_data.shape[0], x_data.shape[1], x_data.shape[2]

        # Individual dead pixels
        missing_mask_2d = np.random.binomial(
            1, p_single, size=(batch, height, width)
        ).astype(np.float32)

        # Clusters (small rectangles)
        n_clusters = int(p_cluster * height * width)
        for _ in range(n_clusters):
            b = np.random.randint(0, batch)
            r, c = np.random.randint(0, height), np.random.randint(0, width)
            sz = np.random.randint(2, cluster_size + 1)
            missing_mask_2d[b, r : r + sz, c : c + sz] = 1

        # Limit to foreground
        missing_mask_limited = missing_mask_2d * foreground_mask
        missing_mask = np.stack(
            (missing_mask_limited,) * num_channels, axis=-1
        ).astype(np.float32)

        x_data_md = self._apply_mask(x_data, missing_mask)

        return x_data, x_data_md, missing_mask

    # ────────────────────────────────────────────────────────────────────────
    # MAR: Missing At Random (depends on observed covariates)
    # ────────────────────────────────────────────────────────────────────────

    def generate_mar_stripes(
        self, x_data: np.ndarray, frac_bad_cols: float = 0.04, stripe_width: int = 1
    ) -> tuple:
        """
        Simulate detector stripe artifacts (vertical lines).

        Models detector column miscalibration causing vertical stripe artifacts.
        Missingness depends on spatial position (observed covariate).
        Morphology: continuous vertical lines in specific columns.

        **Recommended for:**
        - Multi-detector CT artifacts
        - Detector miscalibration
        - Acquisition system defects

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)
        frac_bad_cols : float, optional
            Fraction of columns with stripes. Default: 0.04 (4%)
        stripe_width : int, optional
            Width of each stripe in pixels. Default: 1

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with stripe artifacts (NaN values)
        mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.1)
        >>> original, missing, mask = amputer.generate_mar_stripes(
        ...     image, frac_bad_cols=0.05
        ... )
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        batch, height, width = x_data.shape[0], x_data.shape[1], x_data.shape[2]

        missing_mask_2d = np.zeros((batch, height, width), dtype=np.float32)
        n_bad = max(1, int(frac_bad_cols * width))
        bad_cols = np.random.choice(width, size=n_bad, replace=False)

        for c in bad_cols:
            missing_mask_2d[:, :, c : c + stripe_width] = 1

        # Limit to foreground
        missing_mask_limited = missing_mask_2d * foreground_mask

        x_data_md = self._apply_mask(x_data,missing_mask_limited)

        return x_data, x_data_md, missing_mask_limited

    def generate_mar_truncation(
        self,
        x_data: np.ndarray,
        side: str = "bottom",
        frac: float = 0.08,
        smooth_sigma: float = 5,
    ) -> tuple:
        """
        Simulate field-of-view (FOV) truncation at image borders.

        Models acquisition cutoff at image edges, common in limited-angle tomography.
        Missingness depends on geometric position (distance from edge) — observed covariate.
        Morphology: band with increasing probability toward the edge, smoothly tapered.

        **Recommended for:**
        - Limited-angle tomography
        - FOV truncation
        - Incomplete scan simulation

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)
        side : str, optional
            Which edge to truncate: 'bottom' or 'right'. Default: 'bottom'
        frac : float, optional
            Fraction of image affected by truncation. Default: 0.08
        smooth_sigma : float, optional
            Gaussian smoothing sigma for soft transition. Default: 5

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with FOV truncation (NaN values at edges)
        mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.1)
        >>> original, missing, mask = amputer.generate_mar_truncation(
        ...     image, side='right', frac=0.1
        ... )
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        batch, height, width = x_data.shape[0], x_data.shape[1], x_data.shape[2]

        missing_mask_2d = np.zeros((batch, height, width), dtype=np.float32)

        for b in range(batch):
            dist = np.zeros((height, width), dtype=np.float32)
            if side == "bottom":
                for i in range(height):
                    dist[i, :] = max(0, i - (height * (1 - frac))) / (height * frac + 1e-6)
            elif side == "right":
                for j in range(width):
                    dist[:, j] = max(0, j - (width * (1 - frac))) / (width * frac + 1e-6)

            prob = np.clip(dist, 0, 1)
            prob = gaussian_filter(prob, sigma=smooth_sigma)
            missing_mask_2d[b] = np.random.binomial(1, prob).astype(np.float32)

        # Limit to foreground
        missing_mask_limited = missing_mask_2d * foreground_mask


        x_data_md = self._apply_mask(x_data, missing_mask_limited)

        return x_data, x_data_md, missing_mask_limited

    def generate_mar_compression_plate(
        self,
        x_data: np.ndarray,
        thickness_map: np.ndarray,
        threshold_frac: float = 0.15,
        sigma: float = 8,
    ) -> tuple:
        """
        Simulate compression plate artifact at thin regions.

        Models breast compression artifacts at plate edges (mammography-specific).
        Missingness depends on tissue thickness map (observed covariate).
        Morphology: gradient at edges of compressed regions.

        **Recommended for:**
        - Mammography simulation
        - Tissue compression artifacts
        - Multi-modal imaging

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)
        thickness_map : np.ndarray
            Tissue thickness map normalized to [0, 1].
            0 = thin (edge), 1 = thick (center)
        threshold_frac : float, optional
            Threshold fraction for artifact boundary. Default: 0.15
        sigma : float, optional
            Gaussian smoothing sigma. Default: 8

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with compression plate artifacts (NaN values)
        mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.1)
        >>> thickness = np.linspace(0, 1, 512).reshape(512, 1)
        >>> original, missing, mask = amputer.generate_mar_compression_plate(
        ...     image, thickness_map=thickness
        ... )
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        num_channels = x_data.shape[-1]
        batch, height, width = x_data.shape[0], x_data.shape[1], x_data.shape[2]

        # Ensure thickness_map has compatible shape
        if thickness_map.shape != (batch, height, width):
            thickness_map = np.broadcast_to(thickness_map, (batch, height, width))

        thresh = threshold_frac
        prob = np.clip(1.0 - thickness_map / (thresh + 1e-8), 0, 1) ** 2

        missing_mask_2d = np.zeros((batch, height, width), dtype=np.float32)
        for b in range(batch):
            prob_smooth = gaussian_filter(prob[b], sigma=sigma)
            missing_mask_2d[b] = np.random.binomial(1, prob_smooth).astype(np.float32)

        # Limit to foreground
        missing_mask_limited = missing_mask_2d * foreground_mask
        missing_mask = np.stack(
            (missing_mask_limited,) * num_channels, axis=-1
        ).astype(np.float32)

        x_data_md = self._apply_mask(x_data, missing_mask)

        return x_data, x_data_md, missing_mask

    # ────────────────────────────────────────────────────────────────────────
    # MNAR: Missing Not At Random (depends on unobserved data)
    # ────────────────────────────────────────────────────────────────────────

    def generate_mnar_intensity(self, x_data: np.ndarray) -> tuple:
        """
        Generate missing pixels based on local intensity (MNAR mechanism).

        Brighter pixels (higher intensity) have higher probability of being missing.
        This models systematic signal loss in dense tissue or overexposed regions.
        Missingness depends on the pixel's own value (unobserved if missing).

        **Recommended for:**
        - Saturation/overexposure modeling
        - Dense tissue signal loss
        - Beam hardening artifacts

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with intensity-dependent missing values (NaN)
        mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.15)
        >>> original, missing, mask = amputer.generate_mnar_intensity(image)
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        num_channels = x_data.shape[-1]
        batch, height, width = x_data.shape[0], x_data.shape[1], x_data.shape[2]

        # Average across channels to get grayscale
        grayscale = x_data.mean(axis=-1)  # Shape: (batch, height, width)

        # Normalize intensities to [0, 1]
        grayscale_flat = grayscale[foreground_mask]
        if len(grayscale_flat) > 0:
            gmin, gmax = grayscale_flat.min(), grayscale_flat.max()
            grayscale_norm = (grayscale - gmin) / (gmax - gmin + 1e-8)
        else:
            grayscale_norm = grayscale

        # Probability increases with intensity, scaled by missing_rate
        prob_missing = grayscale_norm * self.missing_rate

        missing_mask = np.random.binomial(1, prob_missing).astype(np.float32)
        missing_mask_limited = missing_mask * foreground_mask

        missing_mask = np.stack(
            (missing_mask_limited,) * num_channels, axis=-1
        ).astype(np.float32)

        x_data_md = self._apply_mask(x_data, missing_mask)

        return x_data, x_data_md, missing_mask

    def generate_mnar_saturation(
        self,
        x_data: np.ndarray,
        alpha: float = 8.0,
        threshold: float = 0.65,
        sigma: float = 4,
    ) -> tuple:
        """
        Simulate saturation dropout in high-intensity regions.

        High-intensity pixels (dense tissue) have much higher probability of being missing.
        Models sensor saturation or signal clipping. Missingness depends on pixel intensity
        itself (unobserved mechanism). Morphology: irregular blobs in brightest regions.

        **Recommended for:**
        - Sensor saturation
        - Dense tissue dropout
        - Nonlinear detector response

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)
        alpha : float, optional
            Steepness of sigmoid transition. Default: 8.0 (sharp)
        threshold : float, optional
            Intensity threshold (normalized [0,1]). Default: 0.65
        sigma : float, optional
            Gaussian smoothing for spatial coherence. Default: 4

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with saturation dropouts (NaN values)
        mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.1)
        >>> original, missing, mask = amputer.generate_mnar_saturation(
        ...     image, alpha=10.0, threshold=0.7
        ... )
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        num_channels = x_data.shape[-1]
        batch, height, width = x_data.shape[0], x_data.shape[1], x_data.shape[2]

        grayscale = x_data.mean(axis=-1)

        missing_mask_2d = np.zeros((batch, height, width), dtype=np.float32)
        for b in range(batch):
            img_smooth = gaussian_filter(grayscale[b], sigma=sigma)
            # Sigmoid: P(missing | intensity) = 1 / (1 + exp(-alpha*(intensity - threshold)))
            prob = 1.0 / (1.0 + np.exp(-alpha * (img_smooth - threshold)))
            missing_mask_2d[b] = np.random.binomial(1, prob).astype(np.float32)

        # Limit to foreground
        missing_mask_limited = missing_mask_2d * foreground_mask
        missing_mask = np.stack(
            (missing_mask_limited,) * num_channels, axis=-1
        ).astype(np.float32)

        x_data_md = self._apply_mask(x_data, missing_mask)

        return x_data, x_data_md, missing_mask

    def generate_mnar_motion_edges(
        self,
        x_data: np.ndarray,
        grad_threshold: float = 0.15,
        alpha: float = 6.0,
        sigma: float = 2,
    ) -> tuple:
        """
        Simulate motion blur at anatomical edges (high-gradient regions).

        Motion artifacts are more severe at structural boundaries where spatial
        gradients are high. Missingness depends on local image gradients (unobserved
        mechanism if pixel is missing). Morphology: streaks along edges.

        **Recommended for:**
        - Motion artifact simulation
        - Boundary blur
        - Breathing/cardiac motion

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)
        grad_threshold : float, optional
            Gradient magnitude threshold. Default: 0.15
        alpha : float, optional
            Sigmoid steepness. Default: 6.0
        sigma : float, optional
            Gaussian smoothing sigma. Default: 2

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with motion blur artifacts (NaN values)
        mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.1)
        >>> original, missing, mask = amputer.generate_mnar_motion_edges(
        ...     image, grad_threshold=0.2, alpha=8.0
        ... )
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        num_channels = x_data.shape[-1]
        batch, height, width = x_data.shape[0], x_data.shape[1], x_data.shape[2]

        grayscale = x_data.mean(axis=-1)

        missing_mask_2d = np.zeros((batch, height, width), dtype=np.float32)
        for b in range(batch):
            # Gradient via Sobel
            gx = np.gradient(grayscale[b], axis=1)
            gy = np.gradient(grayscale[b], axis=0)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_norm = grad_mag / (grad_mag.max() + 1e-8)
            grad_smooth = gaussian_filter(grad_norm, sigma=sigma)

            # P(missing | gradient) = sigmoid
            prob = 1.0 / (1.0 + np.exp(-alpha * (grad_smooth - grad_threshold)))
            missing_mask_2d[b] = np.random.binomial(1, prob).astype(np.float32)

        # Limit to foreground
        missing_mask_limited = missing_mask_2d * foreground_mask
        missing_mask = np.stack(
            (missing_mask_limited,) * num_channels, axis=-1
        ).astype(np.float32)

        x_data_md = self._apply_mask(x_data, missing_mask)

        return x_data, x_data_md, missing_mask

    def generate_mnar_low_snr(
        self,
        x_data: np.ndarray,
        density_map: np.ndarray,
        snr_alpha: float = 5.0,
        density_weight: float = 0.6,
        snr_threshold: float = 0.4,
        sigma: float = 6,
    ) -> tuple:
        """
        Simulate lesion obscuration in low-SNR, high-density regions.

        Lesions with low local contrast in dense tissue are most prone to being missing.
        Combines tissue density and local SNR: both high-density and low-contrast regions
        have higher missing probability. Missingness depends on the signal itself (MNAR).
        Morphology: blobs in dense, low-contrast areas.

        **Recommended for:**
        - Lesion visibility modeling
        - Density-dependent artifacts
        - Diagnostic quality simulation

        Parameters
        ----------
        x_data : np.ndarray
            Input image (2D: H×W, 3D: H×W×C, or batch dimension allowed)
        density_map : np.ndarray
            Tissue density map normalized to [0, 1].
            0 = sparse, 1 = very dense (e.g., glandular tissue)
        snr_alpha : float, optional
            Sigmoid steepness. Default: 5.0
        density_weight : float, optional
            Weight for density vs. SNR in combined score. Default: 0.6
        snr_threshold : float, optional
            SNR threshold. Default: 0.4
        sigma : float, optional
            Gaussian smoothing sigma. Default: 6

        Returns
        -------
        original : np.ndarray
            Normalized image without any modifications
        missing_data : np.ndarray
            Image with low-SNR artifacts (NaN values)
        mask : np.ndarray
            Binary mask (1 = missing, 0 = present)

        Example
        -------
        >>> amputer = ImageDataAmputation(missing_rate=0.15)
        >>> density = np.random.rand(512, 512)
        >>> original, missing, mask = amputer.generate_mnar_low_snr(
        ...     image, density_map=density, density_weight=0.7
        ... )
        """
        x_data, foreground_mask, _ = self._normalize_and_prepare(x_data)
        num_channels = x_data.shape[-1]
        batch, height, width = x_data.shape[0], x_data.shape[1], x_data.shape[2]

        # Ensure density_map has compatible shape
        if density_map.ndim == 2:
            density_map = np.expand_dims(density_map, axis=0)
        if density_map.shape != (batch, height, width):
            density_map = np.broadcast_to(density_map, (batch, height, width))

        grayscale = x_data.mean(axis=-1)

        missing_mask_2d = np.zeros((batch, height, width), dtype=np.float32)
        for b in range(batch):
            # Local contrast: variation around local mean
            local_contrast = gaussian_filter(
                np.abs(grayscale[b] - gaussian_filter(grayscale[b], sigma=20)), sigma=sigma
            )
            local_contrast /= local_contrast.max() + 1e-8
            low_snr = 1.0 - local_contrast  # high value = low SNR

            # Combined: density and low SNR
            combined = density_weight * density_map[b] + (1 - density_weight) * low_snr
            prob = 1.0 / (1.0 + np.exp(-snr_alpha * (combined - snr_threshold)))
            missing_mask_2d[b] = np.random.binomial(1, prob).astype(np.float32)

        # Limit to foreground
        missing_mask_limited = missing_mask_2d * foreground_mask
        missing_mask = np.stack(
            (missing_mask_limited,) * num_channels, axis=-1
        ).astype(np.float32)

        x_data_md = self._apply_mask(x_data, missing_mask)

        return x_data, x_data_md, missing_mask
