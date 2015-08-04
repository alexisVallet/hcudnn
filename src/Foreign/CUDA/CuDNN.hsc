{-|
FFI wrapper around CuDNN.
|-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Foreign.CUDA.CuDNN(
  Status(..)
  , success
  , not_initialized
  , alloc_failed
  , bad_param
  , internal_error
  , invalid_value
  , arch_mismatch
  , mapping_error
  , execution_failed
  , not_supported
  , license_error
  , getErrorString
  , Handle
  , createHandle
  , destroyHandle
  , setStream
  , getStream
  , TensorDescriptor
  , ConvolutionDescriptor
  , PoolingDescriptor
  , FilterDescriptor
  , DataType
  , float
  , double
  , createTensorDescriptor
  , TensorFormat
  , nchw
  , nhwc
  , setTensor4dDescriptor
  , setTensor4dDescriptorEx
  , getTensor4dDescriptor
  , setTensorNdDescriptor
  , getTensorNdDescriptor
  , destroyTensorDescriptor
  , transformTensor
  , AddMode
  , add_image
  , add_same_hw
  , add_feature_map
  , add_same_chw
  , add_same_c
  , add_full_tensor
  , addTensor
  , setTensor
  , ConvolutionMode
  , convolution
  , cross_correlation
  , createFilterDescriptor
  , setFilter4dDescriptor
  , getFilter4dDescriptor
  , setFilterNdDescriptor
  , getFilterNdDescriptor
  , destroyFilterDescriptor
  , createConvolutionDescriptor
  , setConvolution2dDescriptor
  , getConvolution2dDescriptor
  , getConvolution2dForwardOutputDim
  , setConvolutionNdDescriptor
  , getConvolutionNdDescriptor
  , getConvolutionNdForwardOutputDim
  , destroyConvolutionDescriptor
  , ConvolutionFwdPreference
  , convolution_fwd_no_workspace
  , convolution_fwd_prefer_fastest
  , convolution_fwd_specify_workspace_limit
  , ConvolutionFwdAlgo
  , convolution_fwd_algo_implicit_gemm
  , convolution_fwd_algo_implicit_precomp_gemm
  , convolution_fwd_algo_gemm
  , convolution_fwd_algo_direct
  , getConvolutionForwardAlgorithm
  , getConvolutionForwardWorkspaceSize
  , convolutionForward
  , convolutionBackwardBias
  , convolutionBackwardFilter
  , convolutionBackwardData
  , im2Col
  , SoftmaxAlgorithm
  , softmax_fast
  , softmax_accurate
  , SoftmaxMode
  , softmax_mode_instance
  , softmax_mode_channel
  , softmaxForward
  , softmaxBackward
  , PoolingMode
  , pooling_max
  , pooling_average_count_include_padding
  , pooling_average_count_exclude_padding
  , createPoolingDescriptor
  , setPooling2dDescriptor
  , getPooling2dDescriptor
  , setPoolingNdDescriptor
  , getPoolingNdDescriptor
  -- , getPooling2dForwardOutputDim
  -- , getPoolingNdForwardOutputDim
  , destroyPoolingDescriptor
  , poolingForward
  , poolingBackward
  , ActivationMode
  , activation_sigmoid
  , activation_relu
  , activation_tanh
  , activationForward
  , activationBackward
  ) where
import Foreign
import Foreign.C
import Foreign.CUDA.Types

#include <cudnn.h>

-- Version number.
foreign import ccall unsafe "cudnnGetVersion"
  getVersion :: IO CSize

-- CuDNN return codes.
newtype Status = Status {
  unStatus :: CInt
  } deriving (Show, Eq, Storable)

#{enum Status, Status
 , success = CUDNN_STATUS_SUCCESS
 , not_initialized = CUDNN_STATUS_NOT_INITIALIZED
 , alloc_failed = CUDNN_STATUS_ALLOC_FAILED
 , bad_param = CUDNN_STATUS_BAD_PARAM
 , internal_error = CUDNN_STATUS_INTERNAL_ERROR
 , invalid_value = CUDNN_STATUS_INVALID_VALUE
 , arch_mismatch = CUDNN_STATUS_ARCH_MISMATCH
 , mapping_error = CUDNN_STATUS_MAPPING_ERROR
 , execution_failed = CUDNN_STATUS_EXECUTION_FAILED
 , not_supported = CUDNN_STATUS_NOT_SUPPORTED
 , license_error = CUDNN_STATUS_LICENSE_ERROR
 }

foreign import ccall unsafe "cudnnGetErrorString"
  getErrorString :: Status -> IO CString

-- Initializing and destroying CuDNN handles.

-- CuDNN handle is an opaque structure.
newtype Handle = Handle {
  unHandle :: Ptr ()
  } deriving Storable

foreign import ccall unsafe "cudnnCreate"
  createHandle :: Ptr Handle -> IO Status

foreign import ccall unsafe "cudnnDestroy"
  destroyHandle :: Handle -> IO Status

-- Getting and setting stream.
foreign import ccall unsafe "cudnnSetStream"
  setStream :: Handle -> Stream -> IO Status

foreign import ccall unsafe "cudnnGetStream"
  getStream :: Handle -> Ptr Stream -> IO Status

-- Data structures for tensors, convolutions, poolings and filters
-- are also opaque.
newtype TensorDescriptor = TensorDescriptor {
  unTensorDescriptor :: Ptr ()
  } deriving Storable
newtype ConvolutionDescriptor = ConvolutionDescriptor {
  unConvolutionDescriptor :: Ptr ()
  } deriving Storable
newtype PoolingDescriptor = PoolingDescriptor {
  unPoolingDescriptor :: Ptr ()
  } deriving Storable
newtype FilterDescriptor = FilterDescriptor {
  unFilterDescriptor :: Ptr ()
  } deriving Storable

-- Floating point datatypes.
newtype DataType = DataType {
  unDataType :: CInt
  } deriving (Show, Eq, Storable)

#{enum DataType, DataType
 , float = CUDNN_DATA_FLOAT
 , double = CUDNN_DATA_DOUBLE
 }

-- Generic tensor descriptor initialization.
foreign import ccall unsafe "cudnnCreateTensorDescriptor"
  createTensorDescriptor :: Ptr TensorDescriptor -> IO Status

-- Tensor format.
newtype TensorFormat = TensorFormat {
  unTensorFormat :: CInt
  } deriving (Show, Eq, Storable)

#{enum TensorFormat, TensorFormat
 , nchw = CUDNN_TENSOR_NCHW
 , nhwc = CUDNN_TENSOR_NHWC
 }

-- 4d tensor descriptors.
foreign import ccall unsafe "cudnnSetTensor4dDescriptor"
  setTensor4dDescriptor :: TensorDescriptor
                        -> TensorFormat
                        -> DataType
                        -> CInt -- n, batch size
                        -> CInt -- c, number of input feature maps
                        -> CInt -- h, height or rows
                        -> CInt -- w, width or columns
                        -> IO Status

foreign import ccall unsafe "cudnnSetTensor4dDescriptorEx"
  setTensor4dDescriptorEx :: TensorDescriptor
                          -> DataType
                          -> CInt -- n, batch size
                          -> CInt -- c, number of input feature maps
                          -> CInt -- h, height or rows
                          -> CInt -- w, width or columns
                          -> CInt -- nStride
                          -> CInt -- cStride
                          -> CInt -- hStride
                          -> CInt -- wStride
                          -> IO Status

foreign import ccall unsafe "cudnnGetTensor4dDescriptor"
  getTensor4dDescriptor :: TensorDescriptor
                        -> Ptr DataType
                        -> Ptr CInt -- n
                        -> Ptr CInt -- c
                        -> Ptr CInt -- h
                        -> Ptr CInt -- w
                        -> Ptr CInt -- nStride
                        -> Ptr CInt -- cStride
                        -> Ptr CInt -- hStride
                        -> Ptr CInt -- wStride
                        -> IO Status

foreign import ccall unsafe "cudnnSetTensorNdDescriptor"
  setTensorNdDescriptor :: TensorDescriptor
                        -> DataType
                        -> CInt -- nbDims
                        -> Ptr CInt -- dimensions array
                        -> Ptr CInt -- strides array
                        -> IO Status

foreign import ccall unsafe "cudnnGetTensorNdDescriptor"
  getTensorNdDescriptor :: TensorDescriptor
                        -> CInt -- nbDimsRequested
                        -> Ptr DataType
                        -> Ptr CInt -- nbDims
                        -> Ptr CInt -- dimensions array
                        -> Ptr CInt -- strides array
                        -> IO Status

foreign import ccall unsafe "cudnnDestroyTensorDescriptor"
  destroyTensorDescriptor :: TensorDescriptor -> IO Status

-- Apparently a tensor layout conversion helper?
foreign import ccall unsafe "cudnnTransformTensor"
  transformTensor :: Handle
                  -> Ptr a -- alpha
                  -> TensorDescriptor -- srcDesc
                  -> DevicePtr a -- srcData
                  -> Ptr a -- beta
                  -> TensorDescriptor -- destDesc
                  -> DevicePtr a -- destData
                  -> IO Status

-- Tensor in place bias addition.
newtype AddMode = AddMode {
  unAddMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum AddMode, AddMode
 , add_image = CUDNN_ADD_IMAGE
 , add_same_hw = CUDNN_ADD_SAME_HW
 , add_feature_map = CUDNN_ADD_FEATURE_MAP
 , add_same_chw = CUDNN_ADD_SAME_CHW
 , add_same_c = CUDNN_ADD_SAME_C
 , add_full_tensor = CUDNN_ADD_FULL_TENSOR
 }

foreign import ccall unsafe "cudnnAddTensor"
  addTensor :: Handle
            -> AddMode
            -> Ptr a -- alpha
            -> TensorDescriptor -- biasDesc
            -> DevicePtr a -- biasData
            -> Ptr a -- beta
            -> TensorDescriptor -- srcDestDesc
            -> DevicePtr a -- srcDestData
            -> IO Status

-- Fills tensor with value.
foreign import ccall unsafe "cudnnSetTensor"
  setTensor :: Handle
            -> TensorDescriptor -- srcDestDesc
            -> DevicePtr a -- srcDestData
            -> DevicePtr a -- value
            -> IO Status

-- Convolution mode.
newtype ConvolutionMode = ConvolutionMode {
  unConvolutionMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum ConvolutionMode, ConvolutionMode
 , convolution = CUDNN_CONVOLUTION
 , cross_correlation = CUDNN_CROSS_CORRELATION
 }

-- Filter struct manipulation.
foreign import ccall unsafe "cudnnCreateFilterDescriptor"
  createFilterDescriptor :: Ptr FilterDescriptor -> IO Status

foreign import ccall unsafe "cudnnSetFilter4dDescriptor"
  setFilter4dDescriptor :: FilterDescriptor
                        -> DataType
                        -> CInt -- k number of filters
                        -> CInt -- c number of input channels
                        -> CInt -- h height (number of rows) of each filter
                        -> CInt -- w width (number of columns) of each filter
                        -> IO Status

foreign import ccall unsafe "cudnnGetFilter4dDescriptor"
  getFilter4dDescriptor :: FilterDescriptor
                        -> Ptr DataType
                        -> Ptr CInt -- k number of filters
                        -> Ptr CInt -- c number of input channels
                        -> Ptr CInt -- h height (number of rows) of each filter
                        -> Ptr CInt -- w width (number of columns) of each filter
                        -> IO Status

foreign import ccall unsafe "cudnnSetFilterNdDescriptor"
  setFilterNdDescriptor :: FilterDescriptor
                        -> DataType
                        -> CInt -- nbdims
                        -> Ptr CInt -- filter tensor dimensions array
                        -> IO Status

foreign import ccall unsafe "cudnnGetFilterNdDescriptor"
  getFilterNdDescriptor :: FilterDescriptor
                        -> CInt -- number of requested dimensions
                        -> Ptr DataType
                        -> Ptr CInt -- nbdims
                        -> Ptr CInt -- filter tensor dimensions array
                        -> IO Status

foreign import ccall unsafe "cudnnDestroyFilterDescriptor"
  destroyFilterDescriptor :: FilterDescriptor -> IO Status

-- Convolution descriptor manipulations.
foreign import ccall unsafe "cudnnCreateConvolutionDescriptor"
  createConvolutionDescriptor :: Ptr ConvolutionDescriptor
                              -> IO Status

foreign import ccall unsafe "cudnnSetConvolution2dDescriptor"
  setConvolution2dDescriptor :: ConvolutionDescriptor
                             -> CInt -- pad_h
                             -> CInt -- pad_w
                             -> CInt -- u vertical stride
                             -> CInt -- v horizontal stride
                             -> CInt -- upscalex
                             -> CInt -- upscaley
                             -> ConvolutionMode
                             -> IO Status

foreign import ccall unsafe "cudnnGetConvolution2dDescriptor"
  getConvolution2dDescriptor :: ConvolutionDescriptor
                             -> Ptr CInt -- pad_h
                             -> Ptr CInt -- pad_w
                             -> Ptr CInt -- u
                             -> Ptr CInt -- v
                             -> Ptr CInt -- upscalex
                             -> Ptr CInt -- upscaley
                             -> Ptr ConvolutionMode
                             -> IO Status

foreign import ccall unsafe "cudnnGetConvolution2dForwardOutputDim"
  getConvolution2dForwardOutputDim :: ConvolutionDescriptor
                                   -> TensorDescriptor
                                   -> FilterDescriptor
                                   -> Ptr CInt -- n
                                   -> Ptr CInt -- c
                                   -> Ptr CInt -- h
                                   -> Ptr CInt -- w
                                   -> IO Status

foreign import ccall unsafe "cudnnSetConvolutionNdDescriptor"
  setConvolutionNdDescriptor :: ConvolutionDescriptor
                             -> CInt -- array length nbDims-2 size
                             -> Ptr CInt -- paddings array
                             -> Ptr CInt -- filter strides array
                             -> Ptr CInt -- upscales array
                             -> ConvolutionMode
                             -> IO Status

foreign import ccall unsafe "cudnnGetConvolutionNdDescriptor"
  getConvolutionNdDescriptor :: ConvolutionDescriptor
                             -> CInt -- requested array length
                             -> Ptr CInt -- array length
                             -> Ptr CInt -- paddings array
                             -> Ptr CInt -- strides array
                             -> Ptr CInt -- upscales array
                             -> Ptr ConvolutionMode
                             -> IO Status

foreign import ccall unsafe "cudnnGetConvolutionNdForwardOutputDim"
  getConvolutionNdForwardOutputDim :: ConvolutionDescriptor
                                   -> TensorDescriptor
                                   -> FilterDescriptor
                                   -> CInt -- nbDims
                                   -> Ptr CInt -- tensor output dims
                                   -> IO Status

foreign import ccall unsafe "cudnnDestroyConvolutionDescriptor"
  destroyConvolutionDescriptor :: ConvolutionDescriptor
                               -> IO Status

newtype ConvolutionFwdPreference = ConvolutionFwdPreference {
  unConvolutionForwardPreference :: CInt
  } deriving (Show, Eq, Storable)

#{enum ConvolutionFwdPreference, ConvolutionFwdPreference
 , convolution_fwd_no_workspace = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
 , convolution_fwd_prefer_fastest = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, convolution_fwd_specify_workspace_limit = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
 }

newtype ConvolutionFwdAlgo = ConvolutionFwdAlgo {
  unConvolutionFwdAlgo :: CInt
  } deriving (Show, Eq, Storable)

#{enum ConvolutionFwdAlgo, ConvolutionFwdAlgo
 , convolution_fwd_algo_implicit_gemm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
 , convolution_fwd_algo_implicit_precomp_gemm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
 , convolution_fwd_algo_gemm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM
 , convolution_fwd_algo_direct = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
 }

foreign import ccall unsafe "cudnnGetConvolutionForwardAlgorithm"
  getConvolutionForwardAlgorithm :: Handle
                                 -> TensorDescriptor -- srcDesc
                                 -> FilterDescriptor
                                 -> ConvolutionDescriptor
                                 -> TensorDescriptor -- destDesc
                                 -> ConvolutionFwdPreference
                                 -> CSize -- memory limit in bytes
                                 -> Ptr ConvolutionFwdAlgo
                                 -> IO Status

foreign import ccall unsafe "cudnnGetConvolutionForwardWorkspaceSize"
  getConvolutionForwardWorkspaceSize :: Handle
                                     -> TensorDescriptor -- srcDesc
                                     -> FilterDescriptor
                                     -> ConvolutionDescriptor
                                     -> TensorDescriptor -- dstDesc
                                     -> ConvolutionFwdAlgo
                                     -> Ptr CSize -- size in bytes
                                     -> IO Status

foreign import ccall unsafe "cudnnConvolutionForward"
  convolutionForward :: Handle
                     -> Ptr a -- alpha
                     -> TensorDescriptor -- srcDesc
                     -> DevicePtr a -- srcData
                     -> FilterDescriptor
                     -> DevicePtr a -- filterData
                     -> ConvolutionDescriptor
                     -> ConvolutionFwdAlgo
                     -> DevicePtr Int8 -- workspace
                     -> CSize -- workspace size in bytes
                     -> Ptr a -- beta
                     -> TensorDescriptor -- destDesc
                     -> DevicePtr a -- destData
                     -> IO Status

-- Convolution gradient with regards to the bias.
foreign import ccall unsafe "cudnnConvolutionBackwardBias"
  convolutionBackwardBias :: Handle
                          -> Ptr a -- alpha
                          -> TensorDescriptor -- srcDesc
                          -> DevicePtr a -- srcData
                          -> Ptr a -- beta
                          -> TensorDescriptor -- destDesc
                          -> DevicePtr a -- destData
                          -> IO Status

-- Computes gradient with regards to the filters.
foreign import ccall unsafe "cudnnConvolutionBackwardFilter"
  convolutionBackwardFilter :: Handle
                            -> Ptr a -- alpha
                            -> TensorDescriptor -- srcDesc
                            -> DevicePtr a -- srcData
                            -> TensorDescriptor -- diffDesc
                            -> DevicePtr a -- diffData
                            -> ConvolutionDescriptor
                            -> Ptr a -- beta
                            -> FilterDescriptor -- gradDesc
                            -> DevicePtr a -- gradData
                            -> IO Status

-- Computes gradient with regards to the data.
foreign import ccall unsafe "cudnnConvolutionBackwardData"
  convolutionBackwardData :: Handle
                          -> Ptr a -- alpha
                          -> FilterDescriptor
                          -> DevicePtr a -- filterData
                          -> TensorDescriptor -- diffDesc
                          -> DevicePtr a -- diffData
                          -> ConvolutionDescriptor
                          -> Ptr a -- beta
                          -> TensorDescriptor -- gradDesc
                          -> DevicePtr a -- gradData
                          -> IO Status

foreign import ccall unsafe "cudnnIm2Col"
  im2Col :: Handle
         -> Ptr a -- alpha
         -> TensorDescriptor -- srcDesc
         -> DevicePtr a -- srcData
         -> FilterDescriptor
         -> ConvolutionDescriptor
         -> DevicePtr a -- colBuffer
         -> IO Status

-- Softmax
newtype SoftmaxAlgorithm = SoftmaxAlgorithm {
  unSoftmaxAlgorithm :: CInt
  } deriving (Show, Eq, Storable)

#{enum SoftmaxAlgorithm, SoftmaxAlgorithm
 , softmax_fast = CUDNN_SOFTMAX_FAST
 , softmax_accurate = CUDNN_SOFTMAX_ACCURATE
 }

newtype SoftmaxMode = SoftmaxMode {
  unSoftmaxMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum SoftmaxMode, SoftmaxMode
 , softmax_mode_instance = CUDNN_SOFTMAX_MODE_INSTANCE
 , softmax_mode_channel = CUDNN_SOFTMAX_MODE_CHANNEL
 }

foreign import ccall unsafe "cudnnSoftmaxForward"
  softmaxForward :: Handle
                 -> SoftmaxAlgorithm
                 -> SoftmaxMode
                 -> Ptr a -- alpha
                 -> TensorDescriptor -- srcDesc
                 -> DevicePtr a -- srcData
                 -> Ptr a -- beta
                 -> TensorDescriptor -- destDesc
                 -> DevicePtr a -- destData
                 -> IO Status

foreign import ccall unsafe "cudnnSoftmaxBackward"
  softmaxBackward :: Handle
                  -> SoftmaxAlgorithm
                  -> SoftmaxMode
                  -> Ptr a -- alpha
                  -> TensorDescriptor -- srcDesc
                  -> DevicePtr a -- srcData
                  -> TensorDescriptor -- srcDiffDesc
                  -> DevicePtr a  -- srcDiffData
                  -> Ptr a -- beta
                  -> TensorDescriptor -- destDiffDesc
                  -> DevicePtr a -- destDiffData
                  -> IO Status

-- Pooling.
newtype PoolingMode = PoolingMode {
  unPoolingMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum PoolingMode, PoolingMode
 , pooling_max = CUDNN_POOLING_MAX
 , pooling_average_count_include_padding = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
 , pooling_average_count_exclude_padding = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
 }

foreign import ccall unsafe "cudnnCreatePoolingDescriptor"
  createPoolingDescriptor :: Ptr PoolingDescriptor -> IO Status

foreign import ccall unsafe "cudnnSetPooling2dDescriptor"
  setPooling2dDescriptor :: PoolingDescriptor
                         -> PoolingMode
                         -> CInt -- window height
                         -> CInt -- window width
                         -> CInt -- vertical padding
                         -> CInt -- horizontal padding
                         -> CInt -- vertical stride
                         -> CInt -- horizontal stride
                         -> IO Status

foreign import ccall unsafe "cudnnGetPooling2dDescriptor"
  getPooling2dDescriptor :: PoolingDescriptor
                         -> Ptr PoolingMode
                         -> Ptr CInt -- window height
                         -> Ptr CInt -- window width
                         -> Ptr CInt -- vertical padding
                         -> Ptr CInt -- horizontal padding
                         -> Ptr CInt -- vertical stride
                         -> Ptr CInt -- horizontal stride
                         -> IO Status

foreign import ccall unsafe "cudnnSetPoolingNdDescriptor"
  setPoolingNdDescriptor :: PoolingDescriptor
                         -> PoolingMode
                         -> CInt -- nbDims
                         -> Ptr CInt -- window dimensions array
                         -> Ptr CInt -- paddings array
                         -> Ptr CInt -- strides array
                         -> IO Status

foreign import ccall unsafe "cudnnGetPoolingNdDescriptor"
  getPoolingNdDescriptor :: PoolingDescriptor
                         -> CInt -- nbDimsRequested
                         -> Ptr PoolingMode
                         -> Ptr CInt -- nbDims
                         -> Ptr CInt -- window dimensons array
                         -> Ptr CInt -- paddings array
                         -> Ptr CInt -- strides array
                         -> IO Status

-- These 2 functions, although they have headers in cudnn.h, do not
-- have corresponding symbols in libcudnn.so. Bug in CuDNN?
-- foreign import ccall unsafe "cudnnGetPoolingNdForwardOutputDim"
--   getPoolingNdForwardOutputDim :: PoolingDescriptor
--                                -> TensorDescriptor
--                                -> CInt -- nbDims
--                                -> Ptr CInt -- output tensor dimensions
--                                -> IO Status

-- foreign import ccall unsafe "cudnnGetPooling2dForwardOutputDim"
--   getPooling2dForwardOutputDim :: PoolingDescriptor
--                                -> TensorDescriptor
--                                -> Ptr CInt -- outN
--                                -> Ptr CInt -- outC
--                                -> Ptr CInt -- outH
--                                -> Ptr CInt -- outW
--                                -> IO Status

foreign import ccall unsafe "cudnnDestroyPoolingDescriptor"
  destroyPoolingDescriptor :: PoolingDescriptor -> IO Status

foreign import ccall unsafe "cudnnPoolingForward"
  poolingForward :: Handle
                 -> PoolingDescriptor
                 -> Ptr a -- alpha
                 -> TensorDescriptor -- srcDesc
                 -> DevicePtr a -- srcData
                 -> Ptr a -- beta
                 -> TensorDescriptor -- destDesc
                 -> DevicePtr a -- destData
                 -> IO Status

foreign import ccall unsafe "cudnnPoolingBackward"
  poolingBackward :: Handle
                  -> PoolingDescriptor
                  -> Ptr a -- alpha
                  -> TensorDescriptor -- srcDesc
                  -> DevicePtr a -- srcData
                  -> TensorDescriptor -- srcDiffDesc
                  -> DevicePtr a -- srcDiffData
                  -> TensorDescriptor -- destDesc
                  -> DevicePtr a -- destData
                  -> Ptr a -- beta
                  -> TensorDescriptor -- destDiffDesc
                  -> DevicePtr a -- destDiffData
                  -> IO Status

-- Activation functions.
newtype ActivationMode = ActivationMode {
  unActivationMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum ActivationMode, ActivationMode
 , activation_sigmoid = CUDNN_ACTIVATION_SIGMOID
 , activation_relu = CUDNN_ACTIVATION_RELU
 , activation_tanh = CUDNN_ACTIVATION_TANH
 }

foreign import ccall "cudnnActivationForward"
  activationForward :: Handle
                    -> ActivationMode
                    -> Ptr a -- alpha
                    -> TensorDescriptor -- srcDesc
                    -> DevicePtr a -- srcData
                    -> Ptr a -- beta
                    -> TensorDescriptor -- destDesc
                    -> DevicePtr a -- destData
                    -> IO Status

foreign import ccall "cudnnActivationBackward"
  activationBackward :: Handle
                     -> ActivationMode
                     -> Ptr a -- alpha
                     -> TensorDescriptor -- srcDesc
                     -> DevicePtr a -- srcData
                     -> TensorDescriptor -- srcDiffDesc
                     -> DevicePtr a -- srcDiffData
                     -> TensorDescriptor -- destDesc
                     -> DevicePtr a -- destData
                     -> Ptr a --beta
                     -> TensorDescriptor -- destDiffDesc
                     -> DevicePtr a -- destDiffData
                     -> IO Status
