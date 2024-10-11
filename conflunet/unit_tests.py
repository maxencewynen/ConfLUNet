import unittest

import torch

assert torch.cuda.is_available()
from model import Conv3DBlock, UpConv3DBlock, ConfLUNet, UNet3D
from conflunet.evaluation.utils import BinarizeInstancesd, make_offset_matrices
from postprocess import *
from conflunet.evaluation.metrics import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class TestConv3DBlock(unittest.TestCase):
    def test_output_shape(self):
        # Test output shape with bottleneck=False
        in_channels = 1
        out_channels = 64
        batch_size = 2
        shape = (16, 16, 16)
        conv_block = Conv3DBlock(in_channels=in_channels, out_channels=out_channels)
        input_tensor = torch.randn(batch_size, in_channels, *shape)
        output, _ = conv_block(input_tensor)
        self.assertEqual(output.shape[0], batch_size) # Check batch size
        self.assertEqual(output.shape[1], out_channels)  # Check number of output channels
        self.assertEqual(output.shape[2:], (8, 8, 8))  # Check spatial dimensions

    def test_output_shape_bottleneck(self):
        # Test output shape with bottleneck=True
        in_channels = 1
        bottleneck_channel = 512
        batch_size = 2
        shape = (12, 12, 12)
        conv_block = Conv3DBlock(in_channels=in_channels, out_channels=bottleneck_channel, bottleneck=True)
        input_tensor = torch.randn(batch_size, in_channels, *shape)
        output, _ = conv_block(input_tensor)
        self.assertEqual(output.shape[0], batch_size) # Check batch size
        self.assertEqual(output.shape[1], bottleneck_channel)  # Check number of output channels
        self.assertEqual(output.shape[2:], (12, 12, 12))  # Check spatial dimensions

    def test_residual_shape(self):
        # Test shape of the residual connection
        conv_block = Conv3DBlock(in_channels=1, out_channels=64)
        input_tensor = torch.randn(2, 1, 16, 16, 16)
        _, residual = conv_block(input_tensor)
        self.assertEqual(residual.shape[1], 64)  # Check number of output channels
        self.assertEqual(residual.shape[2:], (16, 16, 16))  # Check spatial dimensions

    def test_pooling(self):
        # Test pooling layer presence
        conv_block = Conv3DBlock(in_channels=3, out_channels=64)
        self.assertTrue(hasattr(conv_block, 'pooling'))  # Check attribute existence

    def test_no_pooling_bottleneck(self):
        # Test no pooling layer when bottleneck=True
        conv_block = Conv3DBlock(in_channels=3, out_channels=64, bottleneck=True)
        self.assertFalse(hasattr(conv_block, 'pooling'))  # Check attribute absence

    def test_forward(self):
        # Test forward pass
        conv_block = Conv3DBlock(in_channels=1, out_channels=64)
        input_tensor = torch.randn(2, 1, 8, 8, 8)
        output, _ = conv_block(input_tensor)
        self.assertTrue(torch.is_tensor(output))  # Check output is tensor
        self.assertEqual(output.shape[0], 2)  # Check batch size
        self.assertEqual(output.shape[1], 64)  # Check number of output channels


class TestUpConv3DBlock(unittest.TestCase):
    def test_output_shape(self):
        # Test output shape with last_layer=False
        in_channels = 512
        res_channels = 256
        batch_size = 2
        shape = (8, 8, 8)
        out_shape = (16, 16, 16)
        upconv_block = UpConv3DBlock(in_channels=in_channels, res_channels=res_channels)
        input_tensor = torch.randn(batch_size, in_channels, *shape)  # Fix input shape
        residual = torch.randn(batch_size, res_channels, *out_shape)  # Fix residual shape
        output = upconv_block(input_tensor, residual)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], in_channels // 2)
        self.assertEqual(output.shape[2:], out_shape)

    def test_output_shape_last_layer(self):
        # Test output shape with last_layer=True
        in_channels = 128
        res_channels = 64
        batch_size = 2
        shape = (8, 8, 8)
        out_shape = (16, 16, 16)
        num_classes = 2
        out_channels = num_classes + 4
        upconv_block = UpConv3DBlock(in_channels=in_channels, res_channels=res_channels, last_layer=True, out_channels=out_channels)
        input_tensor = torch.randn(batch_size, in_channels, *shape)  # Fix input shape
        residual = torch.randn(batch_size, res_channels, *out_shape)  # Fix residual shape
        output = upconv_block(input_tensor, residual)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], out_channels)
        self.assertEqual(output.shape[2:], out_shape)

    def test_residual_connection(self):
        # Test presence of residual connection
        upconv_block = UpConv3DBlock(in_channels=64, res_channels=32)
        self.assertTrue(hasattr(upconv_block, 'upconv1'))  # Check attribute existence

    def test_residual_connection_last_layer(self):
        # Test absence of residual connection when last_layer=True
        in_channels = 128
        res_channels = 64
        num_classes = 2
        out_channels = num_classes + 4
        upconv_block = UpConv3DBlock(in_channels=in_channels, res_channels=res_channels, last_layer=True, out_channels=out_channels)
        self.assertTrue(hasattr(upconv_block, 'upconv1'))


class TestConfLUNet(unittest.TestCase):
    def test_output_shapes(self):
        in_channels = 1
        num_classes = 2
        level_channels = (64, 128, 256)
        bottleneck_channel = 512
        separate_decoders = False
        scale_offsets = 1
        batch_size = 2
        shape = (16, 16, 16)

        conf_lunet = ConfLUNet(in_channels, num_classes, level_channels, bottleneck_channel, separate_decoders, scale_offsets)
        input_tensor = torch.randn(batch_size, in_channels, *shape)
        semantic_out, center_prediction_out, offsets_out = conf_lunet(input_tensor)

        self.assertEqual(semantic_out.shape, (batch_size, num_classes, *shape))
        self.assertEqual(center_prediction_out.shape, (batch_size, 1, *shape))
        self.assertEqual(offsets_out.shape, (batch_size, 3, *shape))

    def test_separate_decoders(self):
        in_channels = 1
        num_classes = 2
        level_channels = (64, 128, 256)
        bottleneck_channel = 512
        separate_decoders = True
        scale_offsets = 1
        batch_size = 2
        shape = (16, 16, 16)

        conf_lunet = ConfLUNet(in_channels, num_classes, level_channels, bottleneck_channel, separate_decoders,
                               scale_offsets)
        input_tensor = torch.randn(batch_size, in_channels, *shape)
        semantic_out, center_prediction_out, offsets_out = conf_lunet(input_tensor)

        self.assertEqual(semantic_out.shape, (batch_size, num_classes, *shape))
        self.assertEqual(center_prediction_out.shape, (batch_size, 1, *shape))
        self.assertEqual(offsets_out.shape, (batch_size, 3, *shape))

    def test_forward(self):
        in_channels = 1
        num_classes = 2
        level_channels = (64, 128, 256)
        bottleneck_channel = 512
        separate_decoders = False
        scale_offsets = 1

        conf_lunet = ConfLUNet(in_channels, num_classes, level_channels, bottleneck_channel, separate_decoders,
                               scale_offsets)
        input_tensor = torch.randn(1, in_channels, 32, 32, 32)  # Fix input shape
        semantic_out, center_prediction_out, offsets_out = conf_lunet(input_tensor)

        self.assertTrue(torch.is_tensor(semantic_out))
        self.assertTrue(torch.is_tensor(center_prediction_out))
        self.assertTrue(torch.is_tensor(offsets_out))

    def test_initialization(self):
        in_channels = 3
        num_classes = 2
        level_channels = (64, 128, 256)
        bottleneck_channel = 512
        separate_decoders = False
        scale_offsets = 1

        conf_lunet = ConfLUNet(in_channels, num_classes, level_channels, bottleneck_channel, separate_decoders,
                               scale_offsets)

        # Check if all necessary attributes are initialized
        self.assertTrue(hasattr(conf_lunet, 'a_block1'))
        self.assertTrue(hasattr(conf_lunet, 'a_block2'))
        self.assertTrue(hasattr(conf_lunet, 'a_block3'))
        self.assertTrue(hasattr(conf_lunet, 'bottleNeck'))
        self.assertTrue(hasattr(conf_lunet, 's_block3'))
        self.assertTrue(hasattr(conf_lunet, 's_block2'))
        self.assertTrue(hasattr(conf_lunet, 's_block1'))
        self.assertTrue(hasattr(conf_lunet, '_init_parameters'))


class TestUNet3D(unittest.TestCase):
    def test_output_shapes(self):
        in_channels = 1
        num_classes = 2
        level_channels = (64, 128, 256)
        bottleneck_channel = 512
        batch_size = 2
        shape = (16, 16, 16)

        unet = UNet3D(in_channels, num_classes, level_channels, bottleneck_channel)
        input_tensor = torch.randn(batch_size, in_channels, *shape)
        semantic_out = unet(input_tensor)

        self.assertEqual(semantic_out.shape, (batch_size, num_classes, *shape))

    def test_forward(self):
        in_channels = 1
        num_classes = 2
        level_channels = (64, 128, 256)
        bottleneck_channel = 512

        unet = UNet3D(in_channels, num_classes, level_channels, bottleneck_channel)
        input_tensor = torch.randn(1, in_channels, 32, 32, 32)  # Fix input shape
        semantic_out = unet(input_tensor)

        self.assertTrue(torch.is_tensor(semantic_out))

    def test_initialization(self):
        in_channels = 3
        num_classes = 2
        level_channels = (64, 128, 256)
        bottleneck_channel = 512

        unet = UNet3D(in_channels, num_classes, level_channels, bottleneck_channel)

        # Check if all necessary attributes are initialized
        self.assertTrue(hasattr(unet, 'a_block1'))
        self.assertTrue(hasattr(unet, 'a_block2'))
        self.assertTrue(hasattr(unet, 'a_block3'))
        self.assertTrue(hasattr(unet, 'bottleNeck'))
        self.assertTrue(hasattr(unet, 's_block3'))
        self.assertTrue(hasattr(unet, 's_block2'))
        self.assertTrue(hasattr(unet, 's_block1'))
        self.assertTrue(hasattr(unet, '_init_parameters'))


class TestComputeHessianEigenvalues(unittest.TestCase):
    def test_output_shape(self):
        # Generate a random 3D image
        image = np.random.rand(32, 32, 32)
        # Compute eigenvalues of Hessian matrix
        eigenvalues = compute_hessian_eigenvalues(image)
        # Check if the output has the correct shape
        self.assertEqual(eigenvalues.shape, (3, 32, 32, 32))  # Assuming the image is 3D

    def test_output_dtype(self):
        # Generate a random 3D image
        image = np.random.rand(32, 32, 32)
        # Compute eigenvalues of Hessian matrix
        eigenvalues = compute_hessian_eigenvalues(image)
        # Check if the output has the correct data type
        self.assertTrue(np.issubdtype(eigenvalues.dtype, np.floating))

    def test_input_shape(self):
        # Generate a random 3D image
        image = np.random.rand(32, 32, 32)
        # Compute eigenvalues of Hessian matrix
        eigenvalues = compute_hessian_eigenvalues(image)
        # Check if the output shape matches the input shape
        self.assertEqual(eigenvalues.shape[1:], image.shape)

    def test_zero_input(self):
        # Generate a zero-filled 3D image
        image = np.zeros((32, 32, 32))
        # Compute eigenvalues of Hessian matrix
        eigenvalues = compute_hessian_eigenvalues(image)
        # Check if the output is zero
        self.assertTrue(np.allclose(eigenvalues, np.zeros_like(eigenvalues)))

    def test_inversed_curve_function(self):
        def f(z):
            x, y = z
            return - ((x - 5) ** 2 + (y - 5) ** 2)

        # Create a 10x10 matrix
        image = np.zeros((10, 10))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i, j] = f((i, j))

        # Compute eigenvalues of Hessian matrix
        eigenvalues = compute_hessian_eigenvalues(image)
        # Check if the output shape matches the input shape
        self.assertTrue(eigenvalues[0, 5, 5] < 0)
        self.assertTrue(eigenvalues[1, 5, 5] < 0)


class TestIsTooSmallFunction(unittest.TestCase):
    def test_lesion_is_too_small_volume(self):
        segmentation = np.zeros((10, 10, 10))
        segmentation[1:3, 1:3, 1:3] = 1  # 8 voxels
        self.assertTrue(is_too_small(segmentation, 1, (1, 1, 1), 14, 3))

    def test_lesion_is_not_too_small_volume(self):
        segmentation = np.zeros((10, 10, 10))
        segmentation[1:4, 1:4, 1:4] = 1  # 27 voxels
        self.assertFalse(is_too_small(segmentation, 1, (1, 1, 1), 14, 3))

    def test_lesion_is_too_small_along_an_axis(self):
        segmentation = np.zeros((10, 10, 10))
        segmentation[1:4, 1:4, 1] = 1  # Single layer
        self.assertTrue(is_too_small(segmentation, 1, (1, 1, 1), 14, 3))

    def test_lesion_is_not_too_small_along_any_axis(self):
        segmentation = np.zeros((10, 10, 10))
        segmentation[1:5, 1:5, 1:5] = 1  # Not too small along any axis
        self.assertFalse(is_too_small(segmentation, 1, (1, 1, 1), 14, 3))

    def test_lesion_id_not_present(self):
        segmentation = np.zeros((10, 10, 10))
        self.assertTrue(is_too_small(segmentation, 99, (1, 1, 1), 14, 3),
                        "Lesion ID not present should be considered too small")

    def test_invalid_voxel_size(self):
        segmentation = np.zeros((10, 10, 10))
        with self.assertRaises(AssertionError):
            is_too_small(segmentation, 1, "invalid", 14, 3)

    def test_invalid_voxel_size_length(self):
        segmentation = np.zeros((10, 10, 10))
        with self.assertRaises(AssertionError):
            is_too_small(segmentation, 1, (1, 1), 14, 3)

    def test_2D_segmentation(self):
        segmentation = np.zeros((10, 10))
        segmentation[1:4, 1:4] = 1 # 9 voxels in 2D
        self.assertFalse(is_too_small(segmentation, 1, (1, 1), 5, 3),
                         "2D lesion should not be considered too small if it meets minimum axis requirement")

    def test_lesion_meets_minimum_size_but_too_small_along_axis(self):
        segmentation = np.zeros((10, 10, 10))
        segmentation[1:10, 1:2, 1:2] = 1  # Meets minimum size but too small along two axes
        self.assertTrue(is_too_small(segmentation, 1, (1, 1, 1), 14, 3))

    def test_lesion_fills_entire_segmentation(self):
        segmentation = np.ones((10, 10, 10))
        self.assertFalse(is_too_small(segmentation, 1, (1, 1, 1), 14, 3),
                         "Lesion that fills the entire segmentation should not be considered too small")


class TestPostprocessProbabilitySegmentation(unittest.TestCase):
    def test_output_shape(self):
        # Generate a random probability segmentation
        probability_segmentation = np.random.rand(32, 32, 32)
        # Perform postprocessing
        segmented_mask = postprocess_probability_segmentation(probability_segmentation)
        # Check if the output has the correct shape
        self.assertEqual(segmented_mask.shape, (32, 32, 32))

    def test_output_dtype(self):
        # Generate a random probability segmentation
        probability_segmentation = np.random.rand(32, 32, 32)
        # Perform postprocessing
        segmented_mask = postprocess_probability_segmentation(probability_segmentation)
        # Check if the output has the correct data type
        self.assertTrue(np.issubdtype(segmented_mask.dtype, np.unsignedinteger)
)

    def test_zero_input(self):
        # Generate a zero-filled probability segmentation
        probability_segmentation = np.zeros((32, 32, 32))
        # Perform postprocessing
        segmented_mask = postprocess_probability_segmentation(probability_segmentation)
        # Check if the output is zero-filled
        self.assertTrue(np.all(segmented_mask == 0))


class TestRemoveSmallLesionsFromInstanceSegmentation(unittest.TestCase):

    def setUp(self):
        # Create sample instance segmentation mask
        self.instance_segmentation = np.zeros((10, 10, 10), dtype=np.uint8)
        self.instance_segmentation[2:5, 2:5, 2:5] = 1  # Add a small lesion
        self.instance_segmentation[7:9, 7:9, 7:9] = 2  # Add a larger lesion

        self.voxel_size = (1, 1, 1)
        self.l_min = 10
        self.minimum_size_along_axis = 0

    def test_output_shape(self):
        # Test if the output shape matches the input shape
        processed_segmentation = remove_small_lesions_from_instance_segmentation(self.instance_segmentation, self.voxel_size, self.l_min, self.minimum_size_along_axis)
        self.assertEqual(processed_segmentation.shape, self.instance_segmentation.shape)

    def test_output_type(self):
        # Test if the output type matches the input type
        processed_segmentation = remove_small_lesions_from_instance_segmentation(self.instance_segmentation, self.voxel_size, self.l_min, self.minimum_size_along_axis)
        self.assertTrue(isinstance(processed_segmentation, np.ndarray))

    def test_voxel_size_type(self):
        # Test if the function raises an error for invalid voxel size type
        with self.assertRaises(AssertionError):
            remove_small_lesions_from_instance_segmentation(self.instance_segmentation, [1, 1, 1], self.l_min, self.minimum_size_along_axis)

    def test_voxel_size_length(self):
        # Test if the function raises an error for invalid voxel size length
        with self.assertRaises(AssertionError):
            remove_small_lesions_from_instance_segmentation(self.instance_segmentation, (1, 1), self.l_min, self.minimum_size_along_axis)

    def test_remove_small_lesions_from_instance_segmentation_unit_voxel_size(self):
        # Test if the small lesion is removed and larger lesion is retained when using a unit voxel size
        processed_segmentation = remove_small_lesions_from_instance_segmentation(self.instance_segmentation, self.voxel_size, self.l_min, self.minimum_size_along_axis)
        self.assertEqual([0, 1], np.unique(processed_segmentation).tolist())

    def test_remove_small_lesions_from_instance_segmentation_small_voxel_size(self):
        # Test if the small lesion is removed and larger lesion is retained when using a small voxel size
        voxel_size = (0.8, 0.8, 0.8)
        processed_segmentation = remove_small_lesions_from_instance_segmentation(self.instance_segmentation, voxel_size, self.l_min, 3)
        self.assertEqual([0], np.unique(processed_segmentation).tolist())

    def test_remove_small_lesions_from_instance_segmentation_large_voxel_size(self):
        # Test if the small lesion is removed and larger lesion is retained when using a large voxel size
        voxel_size = (2, 2, 2)
        processed_segmentation = remove_small_lesions_from_instance_segmentation(self.instance_segmentation, voxel_size, self.l_min, self.minimum_size_along_axis)
        self.assertEqual([0, 1, 2], np.unique(processed_segmentation).tolist())

    def test_remove_small_lesions_from_instance_segmentation(self):
        # Test if the small lesion is removed and larger lesion is retained when using a normal l_min value
        processed_segmentation = remove_small_lesions_from_instance_segmentation(self.instance_segmentation, self.voxel_size, self.l_min, self.minimum_size_along_axis)
        self.assertEqual([0, 1], np.unique(processed_segmentation).tolist())

    def test_remove_small_lesions_from_instance_segmentation_for_lesions_smaller_than_3mm_in_any_axis(self):
        # Test if the small lesion is removed and larger lesion is retained when using a small l_min value
        l_min = 1
        processed_segmentation = remove_small_lesions_from_instance_segmentation(self.instance_segmentation, self.voxel_size, l_min, minimum_size_along_axis=3)
        self.assertEqual([0, 1], np.unique(processed_segmentation).tolist()) # 2 should be removed because it is smaller than 3mm in one direction

    def test_remove_small_lesions_from_instance_segmentation_large_l_min(self):
        # Test if the small lesion is removed and larger lesion is retained when using a large l_min value
        l_min = 100
        processed_segmentation = remove_small_lesions_from_instance_segmentation(self.instance_segmentation, self.voxel_size, l_min)
        self.assertEqual([0], np.unique(processed_segmentation).tolist())


class TestRemoveSmallLesionsFromBinarySegmentation(unittest.TestCase):

    def setUp(self):
        # Create sample binary segmentation mask
        self.binary_segmentation = np.zeros((10, 10, 10), dtype=np.uint8)
        self.binary_segmentation[2:5, 2:5, 2:5] = 1  # Add a small lesion
        self.binary_segmentation[7:9, 7:9, 7:9] = 1  # Add a larger lesion

        self.voxel_size = (1, 1, 1)
        self.l_min = 10

    def test_output_shape(self):
        # Test if the output shape matches the input shape
        processed_segmentation = remove_small_lesions_from_binary_segmentation(self.binary_segmentation, self.voxel_size, self.l_min)
        self.assertEqual(processed_segmentation.shape, self.binary_segmentation.shape)

    def test_output_type(self):
        # Test if the output type matches the input type
        processed_segmentation = remove_small_lesions_from_binary_segmentation(self.binary_segmentation, self.voxel_size, self.l_min)
        self.assertTrue(isinstance(processed_segmentation, np.ndarray))

    def test_voxel_size_type(self):
        # Test if the function raises an error for invalid voxel size type
        with self.assertRaises(AssertionError):
            remove_small_lesions_from_binary_segmentation(self.binary_segmentation, [1, 1, 1], self.l_min)

    def test_voxel_size_length(self):
        # Test if the function raises an error for invalid voxel size length
        with self.assertRaises(AssertionError):
            remove_small_lesions_from_binary_segmentation(self.binary_segmentation, (1, 1), self.l_min)

    def test_segmentation_type(self):
        # Test if the function raises an error for invalid segmentation type
        with self.assertRaises(AssertionError):
            remove_small_lesions_from_binary_segmentation(np.zeros((10, 10, 10)) + 2, self.voxel_size, self.l_min)

    def test_invalid_values_in_mask(self):
        # Test if the function raises an error for invalid segmentation values
        with self.assertRaises(AssertionError):
            remove_small_lesions_from_binary_segmentation(np.ones((10, 10, 10)) + 1, self.voxel_size, self.l_min)

    def test_remove_small_lesions_from_binary_segmentation_unit_voxel_size(self):
        # Test if the small lesion is removed and larger lesion is retained when using a unit voxel size
        processed_segmentation = remove_small_lesions_from_binary_segmentation(self.binary_segmentation, self.voxel_size, self.l_min)
        self.assertEqual([0, 1], np.unique(processed_segmentation).tolist())

    def test_remove_small_lesions_from_binary_segmentation_small_voxel_size(self):
        # Test if the small lesion is removed and larger lesion is retained when using a small voxel size
        voxel_size = (0.8, 0.8, 0.8)
        processed_segmentation = remove_small_lesions_from_binary_segmentation(self.binary_segmentation, voxel_size, self.l_min)
        self.assertEqual([0], np.unique(processed_segmentation).tolist())

    def test_remove_small_lesions_from_binary_segmentation_for_lesions_smaller_than_3mm_in_any_axis_voxel_size(self):
        # Test if the small lesion is removed and larger lesion is retained when using a large voxel size
        voxel_size = (5, 5, 5)
        processed_segmentation = remove_small_lesions_from_binary_segmentation(self.binary_segmentation, voxel_size, self.l_min)
        self.assertEqual([0, 1], np.unique(processed_segmentation).tolist()) # One of the two should be removed because it is smaller than 3mm in one direction

    def test_remove_small_lesions_from_binary_segmentation(self):
        # Test if the small lesion is removed and larger lesion is retained when using a normal l_min value
        processed_segmentation = remove_small_lesions_from_binary_segmentation(self.binary_segmentation, self.voxel_size, self.l_min)
        self.assertEqual([0, 1], np.unique(processed_segmentation).tolist())

    def test_remove_small_lesions_from_binary_segmentation_for_lesions_smaller_than_3mm_in_any_axis_l_min(self):
        # Test if the small lesion is removed and larger lesion is retained when using a small l_min value
        l_min = 1
        processed_segmentation = remove_small_lesions_from_binary_segmentation(self.binary_segmentation, self.voxel_size, l_min)
        self.assertEqual([0, 1], np.unique(processed_segmentation).tolist())


class TestFindInstanceCenter(unittest.TestCase):

    def test_threshold_assertion(self):
        """Test whether the threshold assertion works."""
        ctr_hmp = torch.rand(1, 1, 10, 10, 10)
        with self.assertRaises(AssertionError):
            find_instance_center(ctr_hmp, threshold=-0.1)
        with self.assertRaises(AssertionError):
            find_instance_center(ctr_hmp, threshold=1.1)

    def test_batch_size_assertion(self):
        """Test whether the batch size assertion works."""
        ctr_hmp = torch.rand(2, 1, 10, 10, 10)  # Batch size is not 1
        with self.assertRaises(AssertionError):
            find_instance_center(ctr_hmp)

    def test_channel_size_assertion(self):
        """Test whether the channel size assertion works."""
        ctr_hmp = torch.rand(1, 2, 10, 10, 10)  # Channel size is not 1
        with self.assertRaises(AssertionError):
            find_instance_center(ctr_hmp)

    def test_top_k_assertion(self):
        """Test whether the top_k assertion works."""
        ctr_hmp = torch.rand(1, 1, 10, 10, 10)
        with self.assertRaises(AssertionError):
            find_instance_center(ctr_hmp, top_k=0)
        with self.assertRaises(AssertionError):
            find_instance_center(ctr_hmp, top_k=-1)

    def test_dimension_assertion(self):
        """Test whether the dimension assertion works."""
        ctr_hmp = torch.rand(1, 1, 10, 10)  # Missing one dimension
        with self.assertRaises(AssertionError):
            find_instance_center(ctr_hmp)

    def test_nms_assertion(self):
        #Test if pair nms assertion is respected
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        with self.assertRaises(AssertionError):
            find_instance_center(ctr_hmp, nms_kernel=2)
            find_instance_center(ctr_hmp, nms_kernel=4)

    def test_output_shape(self):
        """Test the shape of the output tensor."""
        ctr_hmp = torch.rand(1, 1, 10, 10, 10)
        centers = find_instance_center(ctr_hmp)
        self.assertEqual(len(centers.size()), 2)
        self.assertEqual(centers.size(1), 3)  # Each center should have (x, y, z) coordinates

    def test_output_type(self):
        """Test the type of the output tensor."""
        ctr_hmp = torch.rand(1, 1, 10, 10, 10)
        centers = find_instance_center(ctr_hmp)
        self.assertTrue(isinstance(centers, torch.Tensor))

    def test_same_device(self):
        """Test if the output tensor is on the same device as the input tensor."""
        ctr_hmp = torch.rand(1, 1, 10, 10, 10)
        centers = find_instance_center(ctr_hmp)
        self.assertEqual(centers.device, ctr_hmp.device)

    def test_output_shape(self):
        """Test if the output shape is correct."""
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        # Simulate a few high-intensity points that should pass the threshold and NMS
        ctr_hmp[0, 0, 5, 5, 5] = 0.9
        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=2)
        self.assertTrue(centers.size(1) == 3) # Each center should have (x, y, z) coordinates
        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=5, top_k=2)
        self.assertTrue(centers.size(1) == 3) # Each center should have (x, y, z) coordinates

    def test_threshold(self):
        # Test if the function works when changing the threshold
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        # Simulate a few high-intensity points that should pass the threshold and NMS
        ctr_hmp[0, 0, 5, 5, 5] = 0.09
        ctr_hmp[0, 0, 3, 3, 3] = 0.11
        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=2)
        self.assertEqual(1, centers.size(0))
        # assert only the coordinate above the threshold is in the output
        self.assertTrue(torch.tensor([3, 3, 3]) in centers)

    def test_with_small_nms_kernel(self):
        # Test if the function works when changing the nms_kernel
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        # Simulate a few high-intensity points that should pass the threshold and NMS
        ctr_hmp[0, 0, 5, 5, 5] = 0.5
        ctr_hmp[0, 0, 3, 3, 3] = 0.5
        ctr_hmp[0, 0, 7, 7, 7] = 0.5
        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=1, top_k=3)
        self.assertEqual(3, centers.size(0))
        # assert every coordinate is in output
        self.assertTrue(torch.tensor([3, 3, 3]) in centers)
        self.assertTrue(torch.tensor([5, 5, 5]) in centers)
        self.assertTrue(torch.tensor([7, 7, 7]) in centers)

    def test_in_general(self):
        # Test if the function works as intended
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        # Simulate a few high-intensity points that should pass the threshold and NMS
        ctr_hmp[0, 0, 5, 5, 5] = 0.5
        ctr_hmp[0, 0, 3, 3, 3] = 0.5
        ctr_hmp[0, 0, 7, 7, 7] = 0.5
        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=3)
        self.assertEqual(3, centers.size(0))
        # assert every coordinate is in output
        self.assertTrue(torch.tensor([3, 3, 3]) in centers)
        self.assertTrue(torch.tensor([5, 5, 5]) in centers)
        self.assertTrue(torch.tensor([7, 7, 7]) in centers)

    def test_with_large_nms_kernel(self):
        # Test if the function works when changing the nms_kernel
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        # Simulate a few high-intensity points that should pass the threshold and NMS
        ctr_hmp[0, 0, 5, 5, 5] = 0.9
        ctr_hmp[0, 0, 3, 3, 3] = 0.5
        ctr_hmp[0, 0, 7, 7, 7] = 0.5
        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=5, top_k=3)
        self.assertEqual(1, centers.size(0))
        # assert only the coordinate with the largest probability is in the output
        self.assertTrue(torch.tensor([5, 5, 5]) in centers)

    def test_topk_1(self):
        # Test if the function works when changing the top_k
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        # Simulate a few high-intensity points that should pass the threshold and NMS
        ctr_hmp[0, 0, 5, 5, 5] = 0.5
        ctr_hmp[0, 0, 3, 3, 3] = 0.4
        ctr_hmp[0, 0, 7, 7, 7] = 0.6
        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=1)
        # assert only the coordinate with the largest probability is in the output
        self.assertTrue(torch.tensor([7, 7, 7]) in centers)

    def test_topk_2(self):
        # Test if the function works when changing the top_k
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        # Simulate a few high-intensity points that should pass the threshold and NMS
        ctr_hmp[0, 0, 5, 5, 5] = 0.5
        ctr_hmp[0, 0, 3, 3, 3] = 0.4
        ctr_hmp[0, 0, 7, 7, 7] = 0.6

        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=2)
        self.assertTrue(torch.tensor([5, 5, 5]) in centers)
        self.assertTrue(torch.tensor([7, 7, 7]) in centers)

    def test_topk_high(self):
        # Test if the function works when changing the top_k
        ctr_hmp = torch.zeros(1, 1, 10, 10, 10)
        # Simulate a few high-intensity points that should pass the threshold and NMS
        ctr_hmp[0, 0, 5, 5, 5] = 0.5
        ctr_hmp[0, 0, 3, 3, 3] = 0.4
        ctr_hmp[0, 0, 7, 7, 7] = 0.6

        centers = find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=100)
        self.assertTrue(torch.tensor([5, 5, 5]) in centers)
        self.assertTrue(torch.tensor([7, 7, 7]) in centers)
        self.assertTrue(torch.tensor([3, 3, 3]) in centers)
        self.assertTrue(centers.size(0) == 3) # make sure only three centers are returned


class TestFindInstanceCentersacls(unittest.TestCase):

    def test_numpy_input(self):
        """Test if the function accepts numpy arrays as input."""
        probability_map = np.random.rand(10, 10, 10)
        semantic_mask = np.random.randint(0, 2, size=(10, 10, 10))
        centers = find_instance_centers_acls(probability_map, semantic_mask)
        self.assertIsInstance(centers, torch.Tensor)

    def test_probability_mask_torch_tensor_input(self):
        """Test if the function accepts PyTorch tensors as input."""
        probability_map = torch.rand(10, 10, 10)
        semantic_mask = torch.randint(0, 2, size=(10, 10, 10))
        centers = find_instance_centers_acls(probability_map, semantic_mask)
        self.assertIsInstance(centers, torch.Tensor)

    def test_probability_mask_numpy_array_input(self):
        """Test if the function accepts PyTorch tensors as input."""
        probability_map = torch.rand(10, 10, 10)
        semantic_mask = np.random.randint(0, 2, size=(10, 10, 10))
        centers = find_instance_centers_acls(probability_map, semantic_mask)
        self.assertIsInstance(centers, torch.Tensor)

    def test_semantic_mask_type_numpy_array(self):
        """Test if the function accepts PyTorch tensors as input."""
        probability_map = torch.rand(10, 10, 10)
        semantic_mask = np.random.randint(0, 2, size=(10, 10, 10))
        centers = find_instance_centers_acls(probability_map, semantic_mask)
        self.assertIsInstance(centers, torch.Tensor)

    def test_semantic_mask_type_torch_tensor(self):
        """Test if the function accepts PyTorch tensors as input."""
        probability_map = torch.rand(10, 10, 10)
        semantic_mask = np.random.randint(0, 2, size=(10, 10, 10))
        centers = find_instance_centers_acls(probability_map, semantic_mask)
        self.assertIsInstance(centers, torch.Tensor)

    def test_output_shape(self):
        """Test if the output shape is as expected."""
        probability_map = np.random.rand(10, 10, 10)
        semantic_mask = np.random.randint(0, 2, size=(10, 10, 10))
        centers = find_instance_centers_acls(probability_map, semantic_mask)
        self.assertEqual(len(centers.size()), 2)
        self.assertEqual(centers.size(1), 3)

    def test_easy_case(self):
        """Test if the function correctly identifies regions with negative eigenvalues."""
        probability_map = np.zeros((10, 10, 10))
        probability_map[3, 3, 3] = 1
        probability_map[5, 5, 5] = 1
        probability_map[7, 7, 7] = 1
        semantic_mask = np.ones((10, 10, 10))
        centers = find_instance_centers_acls(probability_map, semantic_mask)
        self.assertEqual([[3,3,3], [5,5,5], [7,7,7]], sorted(centers.tolist(), key=lambda x: x[0]))

    def test_medium_case(self):
        """Test if the function correctly identifies regions with negative eigenvalues."""
        probability_map = np.zeros((10, 10, 10))
        probability_map[5, 5, 4] = 0.99
        probability_map[5, 5, 5] = 1
        probability_map[5, 5, 6] = 0.99
        probability_map[7, 7, 7] = 1
        semantic_mask = np.ones((10, 10, 10))
        centers = find_instance_centers_acls(probability_map, semantic_mask)
        self.assertEqual([[5,5,5], [7,7,7]], sorted(centers.tolist(), key=lambda x: x[0]))

    def test_device_assignment_cpu(self):
        """Test if the function assigns output to cpu when asked."""
        probability_map = np.random.rand(10, 10, 10)
        semantic_mask = np.random.randint(0, 2, size=(10, 10, 10))
        centers_cpu = find_instance_centers_acls(probability_map, semantic_mask, device='cpu')
        # centers_gpu = find_instance_centers_acls(probability_map, semantic_mask, device='cuda')
        self.assertEqual(centers_cpu.device, torch.device('cpu'))
        # self.assertEqual(centers_gpu.device, torch.device('cuda'))

    def test_device_assignment_gpu(self):
        """Test if the function assigns output to gpu when asked."""
        probability_map = np.random.rand(10, 10, 10)
        semantic_mask = np.random.randint(0, 2, size=(10, 10, 10))
        centers_gpu = find_instance_centers_acls(probability_map, semantic_mask, device='cuda:0')
        self.assertEqual(centers_gpu.device, torch.device('cuda:0'))

    def test_invalid_probability_map_type(self):
        """Test if the function raises an error for invalid probability_map type."""
        with self.assertRaises(AssertionError):
            find_instance_centers_acls(torch.rand(10, 10, 10).tolist(), np.random.randint(0, 2, size=(10, 10, 10)))

    def test_invalid_semantic_map_map_type(self):
        """Test if the function raises an error for invalid semantic_map type."""
        with self.assertRaises(AssertionError):
            find_instance_centers_acls(torch.rand(10, 10, 10), np.random.randint(0, 2, size=(10, 10, 10)).tolist())

    def test_mismatching_input_dimensions_1(self):
        """Test if the function raises an error for mismatching input dimensions."""
        with self.assertRaises(AssertionError):
            find_instance_centers_acls(np.random.rand(10, 10), np.random.randint(0, 2, size=(10, 10, 10)))

    def test_mismatching_input_dimensions_2(self):
        """Test if the function raises an error for mismatching input dimensions."""
        with self.assertRaises(AssertionError):
            find_instance_centers_acls(np.random.rand(10, 10, 10), np.random.randint(0, 2, size=(10, 10)))

    def test_mismatching_input_shapes_1(self):
        """Test if the function raises an error for mismatching input shapes."""
        with self.assertRaises(AssertionError):
            find_instance_centers_acls(np.random.rand(10, 10, 11), np.random.randint(0, 2, size=(10, 10, 10)))

    def test_mismatching_input_shapes_2(self):
        """Test if the function raises an error for mismatching input shapes."""
        with self.assertRaises(AssertionError):
            find_instance_centers_acls(np.random.rand(10, 10, 10), np.random.randint(0, 2, size=(10, 10, 11)))


class TestGroupPixels(unittest.TestCase):

    def setUp(self):
        # Common setup for multiple tests
        self.ctr = torch.tensor([[0, 0, 0], [5, 5, 5]], dtype=torch.float32)
        self.offsets = torch.rand(1, 1, 10, 10, 10)  # N=1, C=3, D=10, H=10, W=10

    def test_batch_size_not_one_raises_value_error(self):
        """Test that a ValueError is raised if the batch size of offsets is not 1."""
        offsets = torch.rand(2, 3, 10, 10, 10)  # Incorrect batch size
        with self.assertRaises(AssertionError):
            group_pixels(self.ctr, offsets)

    def test_output_shape_compute_voting_true(self):
        """Test if the output shape is correct."""
        output, votings = group_pixels(self.ctr, self.offsets, compute_voting=True)
        self.assertEqual((1,) + self.offsets.shape[2:], output.shape)
        self.assertEqual(self.offsets.shape[2:], votings.shape)

    def test_output_shape_compute_voting_false(self):
        """Test if the output shape is correct."""
        output = group_pixels(self.ctr, self.offsets, compute_voting=False)
        self.assertEqual((1,) + self.offsets.shape[2:], output.shape)

    def test_correct_output_type(self):
        """Test output shapes when compute_voting is True."""
        instance_id, votes = group_pixels(self.ctr, self.offsets, compute_voting=True)
        self.assertEqual((1,) + self.offsets.shape[2:], instance_id.shape)
        self.assertIsInstance(votes, torch.Tensor)
        self.assertEqual(self.offsets.shape[2:], votes.shape)

    def test_no_centers_compute_voting_false(self):
        """Test function behavior when no center points are provided."""
        ctr_empty = torch.empty((0, 3), dtype=torch.float32)
        output = group_pixels(ctr_empty, self.offsets, compute_voting=False)
        self.assertTrue(torch.equal(output, torch.zeros(*((1,) + self.offsets.shape[2:]), dtype=torch.int16)))

    def test_no_centers_compute_voting_true(self):
        """Test function behavior when no center points are provided."""
        ctr_empty = torch.empty((0, 3), dtype=torch.float32)
        output, votings = group_pixels(ctr_empty, self.offsets, compute_voting=True)
        self.assertTrue(torch.equal(output, torch.zeros(*((1,) + self.offsets.shape[2:]), dtype=torch.int16)))
        self.assertEqual(self.offsets.shape[2:], votings.shape)

    def test_device_consistency(self):
        """Test if the output is on the same device as the input."""
        self.offsets = self.offsets.to('cuda')
        self.ctr = self.ctr.to('cuda')
        output = group_pixels(self.ctr, self.offsets)
        self.assertTrue(output.is_cuda, "Output tensor is not on CUDA.")

    def test_single_center(self):
        """Test with a single center point."""
        ctr = torch.tensor([[5, 5, 5]], dtype=torch.float32)  # One center point
        offsets = torch.zeros(1, 3, 10, 10, 10)  # Zero offsets
        expected = torch.ones(1, 10, 10, 10, dtype=torch.int16)
        output = group_pixels(ctr, offsets)
        self.assertTrue(torch.equal(output, expected), "Output does not match expected output for a single center.")

    def test_multiple_centers_identical_offsets(self):
        """Test with multiple centers and identical offsets."""
        ctr = torch.tensor([[2, 2, 2], [7, 7, 7]], dtype=torch.float32)
        offsets = torch.zeros(1, 3, 10, 10, 10)  # Zero offsets
        output = group_pixels(ctr, offsets)
        unique_ids = torch.unique(output)
        self.assertEqual(2, len(unique_ids), "Expected 2 unique IDs plus background.")

    def test_offset_direction(self):
        """Test that pixels are assigned to the nearest center considering offsets."""
        ctr = torch.tensor([[0, 0, 0], [9, 9, 9]], dtype=torch.float32)  # Two center points at corners
        offsets = torch.zeros(1, 3, 10, 10, 10)  # Start with zero offsets
        offsets[0, :, 5:, 5:, 5:] = 1  # Modify offsets to "push" center assignment
        output = group_pixels(ctr, offsets)
        # Check if the right bottom corner has a different ID, assuming it gets pushed towards the second center
        self.assertNotEqual(output[0, 0, 0, 0].item(), output[0, 9, 9, 9].item(),
                            "Offsets did not affect center assignment correctly.")

    def test_votes_are_correctly_computed(self):
        """Test if votes are correctly computed."""
        ctr = torch.tensor([[0, 0, 0], [9, 9, 9]], dtype=torch.float32)  # Two center points at
        offsets = torch.zeros(1, 3, 10, 10, 10)
        # Point (1, 0, 0), (0, 1, 0) and (0, 0, 1) coordinates to center (0, 0, 0)
        offsets[0, :, 1, 0, 0] = -1
        offsets[0, :, 0, 1, 0] = -1
        offsets[0, :, 0, 0, 1] = -1
        # Point (8, 9, 9), (9, 8, 9) and (9, 9, 8) coordinates to center (9, 9, 9)
        offsets[0, :, 8, 9, 9] = 1
        offsets[0, :, 9, 8, 9] = 1
        offsets[0, :, 9, 9, 8] = 1

        _, votes = group_pixels(ctr, offsets, compute_voting=True)
        # Check if the votes are correctly computed
        self.assertEqual(4, votes[0, 0, 0].item(), "Votes are not correctly computed.")
        self.assertEqual(4, votes[9, 9, 9].item(), "Votes are not correctly computed.")
        # Check if coordinates that voted for a center have 0 votes
        self.assertEqual(0, votes[0, 0, 1].item(), "Votes are not correctly computed.")
        self.assertEqual(0, votes[0, 1, 0].item(), "Votes are not correctly computed.")
        self.assertEqual(0, votes[1, 0, 0].item(), "Votes are not correctly computed.")
        self.assertEqual(0, votes[9, 9, 8].item(), "Votes are not correctly computed.")
        self.assertEqual(0, votes[9, 8, 9].item(), "Votes are not correctly computed.")
        self.assertEqual(0, votes[8, 9, 9].item(), "Votes are not correctly computed.")
        # Check if coordinates whose offset is 0 have 1 votes
        self.assertEqual(1, votes[1, 1, 0].item(), "Votes are not correctly computed.")
        self.assertEqual(1, votes[1, 0, 1].item(), "Votes are not correctly computed.")
        self.assertEqual(1, votes[0, 1, 1].item(), "Votes are not correctly computed.")
        self.assertEqual(1, votes[8, 8, 9].item(), "Votes are not correctly computed.")
        self.assertEqual(1, votes[8, 9, 8].item(), "Votes are not correctly computed.")
        self.assertEqual(1, votes[9, 8, 8].item(), "Votes are not correctly computed.")
        self.assertTrue(torch.all(votes[2:8, 2:8, 2:8] == 1), "Votes are not correctly computed.")

    def test_edge_case_large_offset_but_only_one_center(self):
        """Test behavior when no pixels are near any centers due to extreme offsets."""
        ctr = torch.tensor([[5, 5, 5]], dtype=torch.float32)  # One center point
        offsets = torch.ones(1, 3, 10, 10, 10) * 100  # Extreme positive offsets
        output = group_pixels(ctr, offsets)
        self.assertTrue(torch.all(output == 1))

        offsets = - torch.ones(1, 3, 10, 10, 10) * 100  # Extreme negative offsets
        output = group_pixels(ctr, offsets)
        self.assertTrue(torch.all(output == 1))


class TestRefineInstanceSegmentation(unittest.TestCase):

    def test_no_change_needed(self):
        """Test with an instance mask where no change is needed."""
        instance_mask = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ])
        l_min = 2
        expected_output = instance_mask.copy()
        output = refine_instance_segmentation(instance_mask, voxel_size=(1, 1), l_min=l_min, minimum_size_along_axis=0)
        np.testing.assert_array_equal(expected_output, output, "Instance mask should not change.")

    def test_remove_small_instances(self):
        """Test removing instances smaller than l_min."""
        instance_mask = np.array([
            [1, 1, 1],
            [2, 0, 0],
            [0, 0, 0]
        ])
        l_min = 2
        expected_output = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])
        output = refine_instance_segmentation(instance_mask, voxel_size=(1, 1), l_min=l_min, minimum_size_along_axis=0)
        np.testing.assert_array_equal(expected_output, output, "Small instances were not removed correctly.")

    def test_one_instance_removed(self):
        """Test with all instances being smaller than l_min."""
        instance_mask = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        l_min = 2
        expected_output = np.zeros_like(instance_mask)
        output = refine_instance_segmentation(instance_mask, voxel_size=(1, 1), l_min=l_min, minimum_size_along_axis=0)
        np.testing.assert_array_equal(expected_output, output, "All instances should have been removed.")

    def test_larger_instance_with_disconnected_parts(self):
        """Test a larger instance that has disconnected parts, some parts are smaller than l_min."""
        instance_mask = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 3, 2, 2],
            [0, 0, 0, 0, 2],
            [3, 3, 3, 0, 0],
            [0, 0, 0, 4, 4]
        ])
        l_min = 2
        # Expect smaller part of instance 3 to be removed, and instance 4 to be kept as is.
        expected_output = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 2],
            [3, 3, 3, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        output = refine_instance_segmentation(instance_mask, voxel_size=(1, 1), l_min=l_min, minimum_size_along_axis=0)
        np.testing.assert_array_equal(output, expected_output,
                                      "Larger instance with disconnected parts was not processed correctly.")

    def test_remove_small_instances(self):

        """Test removing instances smaller than l_min."""
        instance_mask = np.array([
            [1, 1, 1],
            [2, 0, 0],
            [0, 0, 3]
        ])
        l_min = 2
        expected_output = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])
        output = refine_instance_segmentation(instance_mask, voxel_size=(1, 1), l_min=l_min, minimum_size_along_axis=0)
        np.testing.assert_array_equal(expected_output, output, "Small instances were not removed correctly.")

    def test_all_instances_removed(self):
        """Test with all instances being smaller than l_min."""
        instance_mask = np.array([
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0]
        ])
        l_min = 1
        expected_output = np.zeros_like(instance_mask)
        output = refine_instance_segmentation(instance_mask, voxel_size=(1, 1), l_min=l_min, minimum_size_along_axis=0)
        np.testing.assert_array_equal(output, expected_output, "All instances should have been removed.")

    def test_relabel_disconnected_components_same_size(self):
        """Test relabeling disconnected components within the same instance."""
        instance_mask = np.array([
            [1, 1, 0, 0, 2],
            [0, 0, 0, 2, 2],
            [1, 1, 0, 0, 2]
        ])
        l_min = 1
        output = refine_instance_segmentation(instance_mask, voxel_size=(1, 1), l_min=l_min, minimum_size_along_axis=0)
        unique_values = np.unique(output[output != 0])
        np.testing.assert_array_equal([1, 2, 3], unique_values, "Expected 3 unique values.")

    def test_relabel_disconnected_components_different_sizes(self):
        """Test relabeling disconnected components within the same instance."""
        instance_mask = np.array([
            [1, 1, 1, 0, 2],
            [0, 0, 0, 2, 2],
            [1, 1, 0, 0, 2]
        ])
        l_min = 1
        output = refine_instance_segmentation(instance_mask,  voxel_size=(1, 1), l_min=l_min, minimum_size_along_axis=0)
        unique_values = np.unique(output[output != 0])
        np.testing.assert_array_equal([1, 2, 3], unique_values, "Expected 3 unique values.")


class TestPostProcess(unittest.TestCase):
    # TODO: Add tests for the post-process function
    pass


class TestDiceMetric(unittest.TestCase):
    def test_dice_metric_perfect_overlap(self):
        # Test case where ground truth and predictions perfectly overlap
        ground_truth = np.ones((5, 5, 5))
        predictions = np.ones((5, 5, 5))
        dice = dice_metric(ground_truth, predictions)
        self.assertEqual(1.0, dice)

    def test_dice_metric_no_overlap(self):
        # Test case where ground truth and predictions have no overlap
        ground_truth = np.zeros((5, 5, 5))
        predictions = np.ones((5, 5, 5))
        dice = dice_metric(ground_truth, predictions)
        self.assertEqual(0.0, dice)

    def test_dice_metric_partial_overlap(self):
        # Test case where ground truth and predictions partially overlap
        ground_truth = np.zeros((5, 5, 5))
        ground_truth[1:4, 1:4, 1:4] = 1
        predictions = np.zeros((5, 5, 5))
        predictions[2:5, 2:5, 2:5] = 1
        dice = dice_metric(ground_truth, predictions)
        excpected_dice = 2 * np.logical_and(predictions, ground_truth).sum() / (np.sum(ground_truth) + np.sum(predictions))
        self.assertEqual(excpected_dice, dice)

    def test_dice_metric_empty_masks(self):
        # Test case where both ground truth and predictions are empty masks
        ground_truth = np.zeros((5, 5, 5))
        predictions = np.zeros((5, 5, 5))
        dice = dice_metric(ground_truth, predictions)
        self.assertEqual(dice, 1.0)

    def test_dice_metric_mismatched_shapes(self):
        # Test case where ground truth and predictions have mismatched shapes
        ground_truth = np.ones((5, 5, 5))
        predictions = np.ones((4, 4, 4))
        with self.assertRaises(AssertionError):
            dice_metric(ground_truth, predictions)

    def test_dice_metric_invalid_predictions(self):
        # Test case where predictions contain values other than 0 and 1
        ground_truth = np.ones((5, 5, 5))
        predictions = np.full((5, 5, 5), 2)
        with self.assertRaises(AssertionError):
            dice_metric(ground_truth, predictions)

    def test_dice_metric_invalid_ground_truth(self):
        # Test case where ground truth contains values other than 0 and 1
        ground_truth = np.full((5, 5, 5), 2)
        predictions = np.ones((5, 5, 5))
        with self.assertRaises(AssertionError):
            dice_metric(ground_truth, predictions)


class TestIntersectionOverUnion(unittest.TestCase):
    def test_intersection_over_union_perfect_overlap(self):
        # Test case where pred_mask and ref_mask perfectly overlap
        pred_mask = np.ones((5, 5, 5))
        ref_mask = np.ones((5, 5, 5))
        iou = intersection_over_union(pred_mask, ref_mask)
        self.assertEqual(iou, 1.0)

    def test_intersection_over_union_no_overlap(self):
        # Test case where pred_mask and ref_mask have no overlap
        pred_mask = np.zeros((5, 5, 5))
        ref_mask = np.ones((5, 5, 5))
        iou = intersection_over_union(pred_mask, ref_mask)
        self.assertEqual(iou, 0.0)

    def test_intersection_over_union_partial_overlap(self):
        # Test case where ground truth and predictions partially overlap
        ground_truth = np.zeros((5, 5, 5))
        ground_truth[1:4, 1:4, 1:4] = 1
        predictions = np.zeros((5, 5, 5))
        predictions[2:5, 2:5, 2:5] = 1
        excpected_iou = np.logical_and(predictions, ground_truth).sum() / \
                         (np.sum(ground_truth) + np.sum(predictions) - np.logical_and(predictions, ground_truth).sum())
        iou = intersection_over_union(predictions, ground_truth)
        self.assertEqual(excpected_iou, iou)

    def test_intersection_over_union_empty_masks(self):
        # Test case where both pred_mask and ref_mask are empty masks
        pred_mask = np.zeros((5, 5, 5))
        ref_mask = np.zeros((5, 5, 5))
        iou = intersection_over_union(pred_mask, ref_mask)
        self.assertEqual(iou, 0.0)

    def test_intersection_over_union_mismatched_shapes(self):
        # Test case where pred_mask and ref_mask have mismatched shapes
        pred_mask = np.ones((5, 5, 5))
        ref_mask = np.ones((4, 4, 4))
        with self.assertRaises(AssertionError):
            intersection_over_union(pred_mask, ref_mask)

    def test_intersection_over_union_invalid_pred_mask(self):
        # Test case where pred_mask contains values other than 0 and 1
        pred_mask = np.full((5, 5, 5), 2)
        ref_mask = np.ones((5, 5, 5))
        with self.assertRaises(AssertionError):
            intersection_over_union(pred_mask, ref_mask)

    def test_intersection_over_union_invalid_ref_mask(self):
        # Test case where ref_mask contains values other than 0 and 1
        pred_mask = np.ones((5, 5, 5))
        ref_mask = np.full((5, 5, 5), 2)
        with self.assertRaises(AssertionError):
            intersection_over_union(pred_mask, ref_mask)


class TestMatchInstances(unittest.TestCase):
    def test_match_instances_perfect_match(self):
        # Test case where pred and ref perfectly match
        pred = np.ones((5, 5, 5))
        ref = np.ones((5, 5, 5))
        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred, ref)
        self.assertEqual([(1, 1)], matched_pairs)
        self.assertEqual(len(unmatched_pred), 0)
        self.assertEqual(len(unmatched_ref), 0)

    def test_match_instances_no_match(self):
        # Test case where pred and ref have no match
        pred = np.zeros((5, 5, 5))
        ref = np.zeros((5, 5, 5))
        pred[1:4, 1:4, 1:4] = 1
        ref[0, 0, 0] = 1
        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred, ref)
        self.assertEqual(len(matched_pairs), 0)
        self.assertEqual(len(unmatched_pred), 1)
        self.assertEqual(len(unmatched_ref), 1)

    def test_match_instances_partial_match(self):
        # Test case where pred and ref have partial match
        pred = np.zeros((5, 5, 5))
        pred[1:4, 1:4, 1:4] = 1
        ref = np.zeros((5, 5, 5))
        ref[2:5, 2:5, 2:5] = 1
        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred, ref, threshold=0)
        self.assertEqual(len(matched_pairs), 1)
        self.assertEqual(len(unmatched_pred), 0)
        self.assertEqual(len(unmatched_ref), 0)

    def test_match_instances_empty_masks(self):
        # Test case where both pred and ref are empty masks
        pred = np.zeros((5, 5, 5))
        ref = np.zeros((5, 5, 5))
        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred, ref)
        self.assertEqual(len(matched_pairs), 0)
        self.assertEqual(len(unmatched_pred), 0)
        self.assertEqual(len(unmatched_ref), 0)

    def test_match_instances_mismatched_shapes(self):
        # Test case where pred and ref have mismatched shapes
        pred = np.ones((5, 5, 5))
        ref = np.ones((4, 4, 4))
        with self.assertRaises(AssertionError):
            match_instances(pred, ref)

    def test_match_instances_invalid_threshold(self):
        # Test case where threshold is not in [0, 1]
        pred = np.ones((5, 5, 5))
        ref = np.ones((5, 5, 5))
        with self.assertWarns(UserWarning):
            match_instances(pred, ref, threshold=1.5)


class TestFindConfluentLesions(unittest.TestCase):
    def test_single_instance_no_confluence(self):
        # Test case with a single instance and no confluence
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[3:6, 3:6, 3:6] = 1
        self.assertEqual(find_confluent_lesions(instance_segmentation), [])

    def test_multiple_disjoint_instances(self):
        # Test case with multiple disjoint instances, no confluent lesions
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[1:3, 1:3, 1:3] = 1
        instance_segmentation[5:7, 5:7, 5:7] = 2
        self.assertEqual(find_confluent_lesions(instance_segmentation), [])

    def test_confluent_instances(self):
        # Test case with confluent instances
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[1:4, 1:4, 1:4] = 1
        instance_segmentation[3:6, 3:6, 3:6] = 2  # Overlaps with instance 1
        confluent_ids = find_confluent_lesions(instance_segmentation)
        self.assertTrue(1 in confluent_ids or 2 in confluent_ids)

    def test_no_instances(self):
        # Test case with no instances
        instance_segmentation = np.zeros((10, 10, 10))
        self.assertEqual(find_confluent_lesions(instance_segmentation), [])

    def test_all_instances_confluent(self):
        # Test case where all instances are part of a single confluent mass
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[1:9, 1:9, 1:9] = np.random.randint(1, 5, (8, 8, 8))
        self.assertTrue(len(find_confluent_lesions(instance_segmentation)) > 0)

    def test_confluent_with_multiple_components(self):
        # Test case with multiple confluent components
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[1:3, 1:3, 1:3] = 1
        instance_segmentation[2:4, 2:4, 2:4] = 2  # Partial overlap with 1
        instance_segmentation[3:5, 3:5, 3:5] = 3  # Partial overlap with 2
        confluent_ids = find_confluent_lesions(instance_segmentation)
        self.assertTrue(set(confluent_ids) == {1, 2, 3})

    def test_sparse_instances_large_volume(self):
        # Sparse instances across a large volume
        instance_segmentation = np.zeros((30, 30, 30))
        instance_segmentation[1:5, 1:5, 1:5] = 1
        instance_segmentation[20:30, 20:30, 20:30] = 2
        self.assertEqual(find_confluent_lesions(instance_segmentation), [])

    def test_confluence_across_edges(self):
        # Instances should not be confluent across the edges of the volume
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[0:2, 0:10, 0:10] = 1
        instance_segmentation[8:10, 0:10, 0:10] = 2
        self.assertEqual(find_confluent_lesions(instance_segmentation), [])

    def test_one_large_confluent_occupying_entire_volume(self):
        # A single, large confluent lesion occupying almost the entire volume
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation.fill(1)
        instance_segmentation[0, 0, 0] = 0  # Small empty space
        instance_segmentation[5, 5, 5] = 2  # Small different instance
        self.assertEqual(set(find_confluent_lesions(instance_segmentation)), {1,2})

    def test_complex_overlaps(self):
        # Complex overlaps among multiple instances
        instance_segmentation = np.zeros((15, 15, 15))
        instance_segmentation[1:5, 1:5, 1:5] = 1
        instance_segmentation[4:8, 4:8, 4:8] = 2
        instance_segmentation[7:11, 7:11, 7:11] = 3
        instance_segmentation[10:14, 10:14, 10:14] = 4
        confluent_ids = find_confluent_lesions(instance_segmentation)
        self.assertTrue(len(confluent_ids) == 4)

    def test_identical_adjacent_instances(self):
        # Identical adjacent instances without overlap but very close proximity
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[1:4, 1:4, 1:4] = 1
        instance_segmentation[4:7, 4:7, 4:7] = 2  # Adjacent to instance 1
        self.assertEqual(find_confluent_lesions(instance_segmentation), [])

    def test_diagonal_contact(self):
        # Instances in diagonal contact, which should not count as confluent
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[1:3, 1:3, 1:3] = 1
        instance_segmentation[3:5, 3:5, 3:5] = 2  # Diagonally adjacent
        self.assertEqual(find_confluent_lesions(instance_segmentation), [])

    def test_varying_size_instances(self):
        # Instances of varying sizes to test handling of different scales
        instance_segmentation = np.zeros((20, 20, 20))
        instance_segmentation[5:15, 5:15, 5:15] = 2  # Large instance
        instance_segmentation[5, 5, 5] = 1  # Small instance overlapping with a large instance
        self.assertEqual(set(find_confluent_lesions(instance_segmentation)), {1,2})

    def test_fully_nested_instances(self):
        # Fully nested instances (one inside another)
        instance_segmentation = np.zeros((10, 10, 10))
        instance_segmentation[1:9, 1:9, 1:9] = 1  # Larger outer instance
        instance_segmentation[4:6, 4:6, 4:6] = 2  # Smaller instance fully inside the larger instance
        self.assertEqual(set(find_confluent_lesions(instance_segmentation)), {1,2})

    # TODO: to be fixed !!
    # def test_edge_case_all_voxels_labelled(self):
    #     # Edge case where all voxels are labelled as different instances
    #     instance_segmentation = np.arange(1, (10*10*10)+1).reshape((10, 10, 10))
    #     self.assertTrue(len(find_confluent_lesions(instance_segmentation)) > 0)


class TestBinarizeInstancesd(unittest.TestCase):
    def test_out_key_is_in_output(self):
        binarizer = BinarizeInstancesd(keys=["instance_segmentation"], out_key="test1212")
        data = {"instance_segmentation": np.ones((64, 64))}
        transformed_data = binarizer(data)
        self.assertTrue("test1212" in transformed_data.keys())

    def test_binarize_single_key(self):
        binarizer = BinarizeInstancesd(keys=["instance_segmentation"])
        data = {"instance_segmentation": np.ones((64, 64))}
        transformed_data = binarizer(data)
        np.testing.assert_array_equal(transformed_data["label"], np.ones((64, 64), dtype=np.uint8))

    def test_binarize_multiple_keys(self):
        binarizer = BinarizeInstancesd(keys=["image", "mask"], out_key="binarized")
        data = {"image": np.ones((64, 64)), "mask": np.zeros((64, 64))}
        transformed_data = binarizer(data)
        self.assertTrue("binarized_image" in transformed_data)
        self.assertTrue("binarized_mask" in transformed_data)
        np.testing.assert_array_equal(transformed_data["binarized_image"], np.ones((64, 64), dtype=np.uint8))
        np.testing.assert_array_equal(transformed_data["binarized_mask"], np.zeros((64, 64), dtype=np.uint8))

    def test_binarize_no_output_key(self):
        binarizer = BinarizeInstancesd(keys=["image", "mask"], out_key="")
        data = {"image": np.ones((64, 64)), "mask": np.zeros((64, 64))}
        transformed_data = binarizer(data)
        self.assertTrue("image" in transformed_data)  # Output dictionary keys should not change

    def test_binarize_non_binary_values(self):
        binarizer = BinarizeInstancesd(keys=["image"])
        data = {"image": np.random.rand(64, 64)}
        transformed_data = binarizer(data)
        self.assertTrue("label" in transformed_data)
        self.assertTrue(np.all(np.logical_or(transformed_data["label"] == 0, transformed_data["label"] == 1)))

    def test_binarize_negative_values(self):
        binarizer = BinarizeInstancesd(keys=["image"])
        data = {"image": -np.ones((64, 64))}
        with self.assertRaises(AssertionError):
            binarizer(data)


class TestMakeOffsetMatrices(unittest.TestCase):
    def test_basic_functionality(self):
        # Create a simple data array with a single lesion
        data = np.zeros((10, 10, 10))
        data[3:6, 3:6, 3:6] = 1

        # Call the function
        heatmap, offsets = make_offset_matrices(data)

        # Check heatmap shape
        self.assertEqual(heatmap.shape, (1, 10, 10, 10), "Heatmap shape is incorrect")

        # Check offsets shape
        self.assertEqual(offsets.shape, (3, 10, 10, 10), "Offsets shape is incorrect")

        # Check if heatmap values are between 0 and 1
        self.assertTrue(np.all(heatmap >= 0), "Heatmap values are below 0")
        self.assertTrue(np.all(heatmap <= 1), "Heatmap values are above 1")

    def test_empty_data(self):
        # Call the function with an empty data array
        data = np.zeros((0, 0, 0))
        heatmap, offsets = make_offset_matrices(data)

        # Check if heatmap and offsets are empty arrays
        self.assertEqual(heatmap.shape, (1, 0, 0, 0))
        self.assertEqual(offsets.shape, (3, 0, 0, 0))

    def test_non_binary_data(self):
        # Create a non-binary data array
        data = np.ones((10, 10, 10)) * 2

        # Call the function with non-binary data
        heatmap, offsets = make_offset_matrices(data)

        # Check if heatmap values are still between 0 and 1
        self.assertTrue(np.all(heatmap >= 0))
        self.assertTrue(np.all(heatmap <= 1))

    def test_large_data(self):
        # Create a large data array
        data = np.zeros((30, 30, 30))
        data[10:20, 10:20, 10:20] = 1

        # Call the function with large data
        heatmap, offsets = make_offset_matrices(data)

        # Check heatmap shape
        self.assertEqual(heatmap.shape, (1, 30, 30, 30))

        # Check offsets shape
        self.assertEqual(offsets.shape, (3, 30, 30, 30))

    def test_offsets(self):
        # Create a simple data array with a single lesion
        data = np.zeros((10, 10, 10))
        data[3:6, 3:6, 3:6] = 2 # instance id

        # Call the function
        heatmap, offsets = make_offset_matrices(data)

        # Check if offsets are correct
        self.assertTrue(np.all(offsets[:, 4, 4, 4] == 0), "Offsets are incorrect")  # center of mass
        self.assertTrue(np.all(offsets[:, 3, 3, 3] == 1), "Offsets are incorrect")
        self.assertTrue(np.all(offsets[:, 5, 5, 5] == -1), "Offsets are incorrect")
        self.assertTrue(np.all(offsets[:, 6, 6, 6] == 0), "Offsets are incorrect")  # are we sure that's what we want?

    def test_heatmap(self):
        # Create a simple data array with a single lesion
        data = np.zeros((10, 10, 10))
        data[3:6, 3:6, 3:6] = 2  # instance id

        # Call the function
        heatmap, offsets = make_offset_matrices(data)

        # Check if heatmap is correct
        self.assertEqual(np.max(heatmap), heatmap[0, 4, 4, 4])
        self.assertEqual(heatmap[0, 3, 3, 3], heatmap[0, 5, 5, 5])
        self.assertEqual(heatmap[0, 2, 2, 2], heatmap[0, 6, 6, 6])

    def test_heatmap_with_multiple_instances(self):
        # Create a simple data array with a single lesion
        data = np.zeros((10, 10, 10))
        data[3:6, 3:6, 3:6] = 2
        data[7:10, 7:10, 7:10] = 3

        # Call the function
        heatmap, offsets = make_offset_matrices(data)

        # Check if heatmap is correct
        self.assertEqual(np.max(heatmap), heatmap[0, 4, 4, 4])
        self.assertEqual(np.max(heatmap), heatmap[0, 8, 8, 8])

    def test_offsets_with_multiple_instances(self):
        # Create a simple data array with a single lesion
        data = np.zeros((10, 10, 10))
        data[3:6, 3:6, 3:6] = 2
        data[7:10, 7:10, 7:10] = 3

        # Call the function
        heatmap, offsets = make_offset_matrices(data)

        # Check if offsets are correct
        self.assertTrue(np.all(offsets[:, 4, 4, 4] == 0), "Offsets are incorrect")
        self.assertTrue(np.all(offsets[:, 3, 3, 3] == 1), "Offsets are incorrect")
        self.assertTrue(np.all(offsets[:, 5, 5, 5] == -1), "Offsets are incorrect")
        self.assertTrue(np.all(offsets[:, 8, 8, 8] == 0), "Offsets are incorrect")
        self.assertTrue(np.all(offsets[:, 7, 7, 7] == 1), "Offsets are incorrect")
        self.assertTrue(np.all(offsets[:, 9, 9, 9] == -1), "Offsets are incorrect")
        self.assertTrue(np.all(offsets[:, 6, 6, 6] == 0), "Offsets are incorrect")

    def test_swap_instance_ids(self):
        # Create a simple data array with a single lesion
        data1 = np.zeros((10, 10, 10))
        data1[3:6, 3:6, 3:6] = 1
        data1[7:10, 7:10, 7:10] = 2

        # Call the function
        heatmap1, offsets1 = make_offset_matrices(data1)

        # Create a simple data array with a single lesion
        data2 = np.zeros((10, 10, 10))
        data2[3:6, 3:6, 3:6] = 2
        data2[7:10, 7:10, 7:10] = 1

        # Call the function
        heatmap2, offsets2 = make_offset_matrices(data2)

        # Check if output is the same
        self.assertTrue(np.all(heatmap1 == heatmap2))
        self.assertTrue(np.all(offsets1 == offsets2))


if __name__ == '__main__':
    unittest.main()
