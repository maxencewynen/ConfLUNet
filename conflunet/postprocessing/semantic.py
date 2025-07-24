import torch
import numpy as np
from typing import Dict, Tuple
from scipy.spatial.distance import cdist
from scipy.ndimage import label, generate_binary_structure
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.cluster import DBSCAN

from monai.config.type_definitions import NdarrayOrTensor

from conflunet.postprocessing.basic_postprocessor import Postprocessor


class ConnectedComponentsPostprocessor(Postprocessor):
    def __init__(
            self,
            minimum_instance_size: int = 0,
            minimum_size_along_axis: int = 0,
            voxel_spacing: Tuple[float, float, float] = None,
            semantic_threshold: float = 0.5,
            device: torch.device = None,
            verbose : bool = True
    ):
        super(ConnectedComponentsPostprocessor, self).__init__(
            minimum_instance_size=minimum_instance_size,
            minimum_size_along_axis=minimum_size_along_axis,
            voxel_spacing=voxel_spacing,
            semantic_threshold=semantic_threshold,
            name="CC",
            device=device,
            verbose=verbose
        )

    def _postprocess(self, output_dict: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        assert 'semantic_pred_proba' in output_dict.keys(), "output_dict must contain 'semantic_pred_proba'"
        semantic_pred_proba = output_dict['semantic_pred_proba']
        binary_pred = np.squeeze(self._maybe_convert_to_numpy(self.binarize_semantic_probability(semantic_pred_proba)))
        instance_seg_pred = label(binary_pred)[0]

        output_dict['instance_seg_pred'] = self._convert_as(instance_seg_pred, semantic_pred_proba)
        output_dict['semantic_pred_binary'] = self._convert_as(binary_pred, semantic_pred_proba)

        output_dict = self.refine_instance_segmentation(output_dict)

        return output_dict


class ACLSPostprocessor(Postprocessor):
    def __init__(
            self,
            minimum_instance_size: int = 0,
            minimum_size_along_axis: int = 0,
            voxel_spacing: Tuple[float, float, float] = None,
            connectivity: int = 26,
            semantic_threshold: float = 0.5,
            sigma: float = 1.0,
            device: torch.device = None,
            verbose: bool = True
    ):
        """sigma (float): Standard deviation of the Gaussian filter used for the Hessian matrix."""
        super(ACLSPostprocessor, self).__init__(
            minimum_instance_size=minimum_instance_size,
            minimum_size_along_axis=minimum_size_along_axis,
            voxel_spacing=voxel_spacing,
            semantic_threshold=semantic_threshold,
            name="ACLS",
            device=device,
            verbose=verbose
        )
        self.sigma = sigma
        self.connectivity = connectivity

    def compute_hessian_eigenvalues(self, image):
        """
        Compute the eigenvalues of the Hessian matrix of an image.
        Args:
            image (np.ndarray): Input image.
        Returns:
            np.ndarray: Eigenvalues of the Hessian matrix.
        """
        hessian_matrices = hessian_matrix(image, sigma=self.sigma, use_gaussian_derivatives=False)
        eigs = hessian_matrix_eigvals(hessian_matrices)
        return eigs

    def find_instance_centers(
            self,
            probability_map: np.ndarray,
            semantic_mask: np.ndarray,
            device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Computes the instance centers using acls's method
        Arguments:
            probability_map: A numpy.ndarray or a torch.Tensor of shape [W, H, D] of raw probability map output
            semantic_mask: A numpy.ndarray or a torch.Tensor of shape [W, H, D] of raw semantic mask output
            device: A string, the device to use
        Returns:
            A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (x, y, z).
        """
        mask = semantic_mask == 1
        masked_image_data = np.where(mask, probability_map, 0)

        eigenvalues = self.compute_hessian_eigenvalues(masked_image_data)
        instance_centers_mask = np.all(eigenvalues < 0, axis=0)

        instance_clusters, n_clusters = label(instance_centers_mask)

        centers = []
        for c in range(1, n_clusters + 1):
            coords = np.where(instance_clusters == c)
            coords = np.stack(coords, axis=1)
            coords = np.round(np.mean(coords, axis=0)).astype(np.int16)
            centers.append(coords)

        centers = torch.from_numpy(np.stack(centers, axis=0)).short().to(device)

        return centers

    @staticmethod
    def find_nearest_instance_labels(unlabelled_voxels_indices, instance_clusters):
        labelled_voxels_indices = np.transpose(np.array(np.where(instance_clusters > 0)))

        # Initialize an array to store the nearest labels
        nearest_labels = np.zeros(len(unlabelled_voxels_indices), dtype=instance_clusters.dtype)

        # Loop through unlabelled voxels and compute distances incrementally
        for i, voxel_index in enumerate(unlabelled_voxels_indices):
            distances = cdist([voxel_index], labelled_voxels_indices)
            nearest_index = np.argmin(distances)
            nearest_labels[i] = instance_clusters[tuple(labelled_voxels_indices[nearest_index])]

        # Assign the nearest instance labels to unlabelled voxels
        instance_clusters[tuple(unlabelled_voxels_indices.T)] = nearest_labels

        return instance_clusters

    def _postprocess(self, output_dict: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        assert 'semantic_pred_proba' in output_dict.keys(), "output_dict must contain 'semantic_pred_proba'"
        semantic_pred_proba = output_dict['semantic_pred_proba']
        binary_pred = np.squeeze(self._maybe_convert_to_numpy(self.binarize_semantic_probability(semantic_pred_proba)))
        instance_seg_pred = label(binary_pred)[0]
        output_dict['instance_seg_pred'] = self._convert_as(instance_seg_pred, semantic_pred_proba)
        output_dict['semantic_pred_binary'] = self._convert_as(binary_pred, semantic_pred_proba)

        output_dict = self.remove_small_instances(output_dict)
        masked_data = np.where(binary_pred, self._maybe_convert_to_numpy(output_dict['semantic_pred_proba']), 0)

        eigenvalues = self.compute_hessian_eigenvalues(np.squeeze(masked_data))
        instance_centers_mask = np.all(eigenvalues < 0, axis=0)

        if self.connectivity == 6:
            structure = generate_binary_structure(3, 1)
        elif self.connectivity == 18:
            structure = generate_binary_structure(3, 2)
        elif self.connectivity == 26:
            structure = np.ones((3, 3, 3), dtype=bool)
        else:
            raise ValueError("Invalid connectivity value. Use 6, 18 or 26.")

        instance_centers_clusters, n_clusters = label(instance_centers_mask, structure)

        # Identify unlabelled voxels in binary image and assign nearest instance labels
        unlabelled_voxels = np.logical_and(binary_pred == 1, instance_centers_clusters == 0)
        unlabelled_voxels_indices = np.transpose(np.where(unlabelled_voxels))
        
        if len(unlabelled_voxels_indices) > 0:
            instance_seg_pred = self.find_nearest_instance_labels(unlabelled_voxels_indices, instance_centers_clusters)

        output_dict['instance_seg_pred'] = self._convert_as(instance_seg_pred, semantic_pred_proba)
        output_dict['center_pred'] = self._convert_as(instance_centers_clusters, semantic_pred_proba)
        output_dict['semantic_pred_binary'] = self._convert_as(binary_pred, semantic_pred_proba)

        output_dict = self.refine_instance_segmentation(output_dict)

        return output_dict

class ClusterOffsetsPostprocessor(Postprocessor):
    def __init__(
            self,
            minimum_instance_size: int = 0,
            minimum_size_along_axis: int = 0,
            voxel_spacing: Tuple[float, float, float] = None,
            semantic_threshold: float = 0.5,
            eps: float = 0.5,
            min_samples: int = 5,
            device: torch.device = None,
            verbose: bool = True
    ):
        super(ClusterOffsetsPostprocessor, self).__init__(
            minimum_instance_size=minimum_instance_size,
            minimum_size_along_axis=minimum_size_along_axis,
            voxel_spacing=voxel_spacing,
            semantic_threshold=semantic_threshold,
            name="ClusterOffsets",
            device=device,
            verbose=verbose
        )
        self.eps = eps
        self.min_samples = min_samples

    def compute_final_coordinates(self, offsets, voxel_spacing=None):
        """
        offsets: np.ndarray of shape (3, H, W, D)
        Returns: final_coords of shape (N, 3), where N = H*W*D
        """
        voxel_spacing = self.voxel_spacing if voxel_spacing is None else voxel_spacing
        if voxel_spacing is None:
            raise ValueError("voxel_spacing must be provided or set in the postprocessor")
        _, H, W, D = offsets.shape
        x, y, z = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
        x = x.astype(float) * voxel_spacing[0]
        y = y.astype(float) * voxel_spacing[1]
        z = z.astype(float) * voxel_spacing[2]
        coords = np.stack([x, y, z], axis=0).astype(float)
        final_coords = coords + offsets
        final_coords = final_coords.reshape(3, -1).T  # (N, 3)
        return final_coords

    @staticmethod
    def cluster_voxels(final_coords, eps, min_samples):
        """
        final_coords: np.ndarray of shape (N, 3)
        Returns: labels of shape (N,), where each value is a cluster label
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit(final_coords)
        return clustering.labels_

    @staticmethod
    def labels_to_image(labels, lesion_mask):
        """
        labels: np.ndarray of shape (N,), cluster labels for lesion voxels only
        lesion_mask: np.ndarray of shape (H, W, D), boolean mask of lesion voxels
        Returns: label_image of shape (H, W, D), 0 for non-lesion, cluster label for lesion voxels
        """
        label_image = np.zeros(lesion_mask.shape, dtype=labels.dtype)
        label_image[lesion_mask] = labels + 1  # add 1 to avoid DBSCAN's -1 noise label being mapped to valid voxels
        return label_image

    def _postprocess(self, output_dict: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        assert 'semantic_pred_proba' in output_dict.keys(), "output_dict must contain 'semantic_pred_proba'"
        assert 'offsets' in output_dict.keys(), "output_dict must contain 'offsets'"

        semantic_pred_proba = output_dict['semantic_pred_proba']
        binary_pred = np.squeeze(self._maybe_convert_to_numpy(self.binarize_semantic_probability(semantic_pred_proba)))

        voxel_spacing = output_dict.get('properties', None)
        voxel_spacing = voxel_spacing['sitk_stuff']['spacing'] if voxel_spacing is not None else self.voxel_spacing
        if voxel_spacing is None:
            raise ValueError("voxel_spacing must be provided or set in the postprocessor")

        final_coords = self.compute_final_coordinates(output_dict['offsets'], voxel_spacing=voxel_spacing)
        final_coords_lesion = final_coords[(binary_pred == 1).flatten()]

        labels = self.cluster_voxels(final_coords_lesion, eps=self.eps, min_samples=self.min_samples)
        instance_seg_pred = self.labels_to_image(labels, binary_pred == 1)

        output_dict['instance_seg_pred'] = self._convert_as(instance_seg_pred, semantic_pred_proba)
        output_dict['semantic_pred_binary'] = self._convert_as(binary_pred, semantic_pred_proba)

        output_dict = self.refine_instance_segmentation(output_dict)

        return output_dict

