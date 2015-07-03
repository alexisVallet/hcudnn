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
                  -> Ptr () -- alpha
                  -> TensorDescriptor -- srcDesc
                  -> Ptr () -- srcData
                  -> Ptr () -- beta
                  -> TensorDescriptor -- destDesc
                  -> Ptr () -- destData
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
            -> Ptr () -- alpha
            -> TensorDescriptor -- biasDesc
            -> Ptr () -- biasData
            -> Ptr () -- beta
            -> TensorDescriptor -- srcDestDesc
            -> Ptr () -- srcDestData
            -> IO Status

-- Fills tensor with value.
foreign import ccall unsafe "cudnnSetTensor"
  setTensor :: Handle
            -> TensorDescriptor -- srcDestDesc
            -> Ptr () -- srcDestData
            -> Ptr () -- value
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
